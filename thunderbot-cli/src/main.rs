use anyhow::Result;
use clap::{Parser, Subcommand};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

use thunderbot_core::{
    AgentEvent, AgentLoop, Message, Plugin, Role, SessionManager, Tool, ToolDefinition, ToolRegistry
};
use crate::models::RoutingRegistry;

mod models;
use thunderbot_tools::{BashTool, ListFilesTool, ReadFileTool, WriteFileTool};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Starts a conversation session with the bot
    Chat {
        /// The session ID
        #[arg(short, long, default_value = "default")]
        session_id: String,

        /// The model to use (overrides configured default)
        #[arg(short = 'M', long)]
        model: Option<String>,

        /// The initial message to send to the bot (optional, starts REPL if not provided)
        #[arg(short = 'm', long)]
        message: Option<String>,
    },
    /// Runs a tool directly for testing
    RunTool {
        /// The name of the tool
        tool: String,

        /// JSON arguments for the tool
        args: String,
    },
    /// Interactively configures LLM provider and default model
    Onboard,
}


#[derive(Debug, Serialize, Deserialize, Default)]
struct Config {
    default_model: Option<String>,
}

impl Config {
    fn config_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", "thunderbot")
            .map(|dirs| dirs.config_dir().join("config.json"))
    }

    fn load() -> Self {
        if let Some(path) = Self::config_path() {
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Ok(config) = serde_json::from_str(&content) {
                        return config;
                    }
                }
            }
        }
        Self::default()
    }

    fn save(&self) -> anyhow::Result<()> {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string_pretty(self)?;
            std::fs::write(path, content)?;
        }
        Ok(())
    }
}

// Simple Tool Registry
struct SimpleToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl SimpleToolRegistry {
    fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }
}

#[async_trait]
impl ToolRegistry for SimpleToolRegistry {
    fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.definition().name.clone(), tool);
    }

    async fn execute_tool(&self, name: &str, args: Value) -> anyhow::Result<Value> {
        if let Some(tool) = self.tools.get(name) {
            tool.execute(args).await
        } else {
            Err(anyhow::anyhow!("Tool not found: {}", name))
        }
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }
}

// Simple Session Manager
struct SimpleSessionManager {
    sessions: RwLock<HashMap<String, Vec<Message>>>,
}

impl SimpleSessionManager {
    fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl SessionManager for SimpleSessionManager {
    async fn load_session(&self, session_id: &str) -> anyhow::Result<Vec<Message>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.get(session_id).cloned().unwrap_or_default())
    }

    async fn append_message(&self, session_id: &str, message: &Message) -> anyhow::Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.entry(session_id.to_string()).or_default();
        session.push(message.clone());
        Ok(())
    }

    async fn compact_context(&self, _session_id: &str) -> anyhow::Result<()> {
        Ok(()) // No-op for now
    }
}

// Simple Plugin for logging events
struct LoggingPlugin;

#[async_trait]
impl Plugin for LoggingPlugin {
    async fn on_event(&self, event: &AgentEvent) -> anyhow::Result<()> {
        match event {
            AgentEvent::LoopStarted => println!("[Plugin] Agent loop started"),
            AgentEvent::CompletionGenerated(_) => println!("[Plugin] Completion generated"),
            AgentEvent::ToolExecutionStarted(call) => println!("[Plugin] Tool execution started: {}", call.name),
            AgentEvent::ToolExecutionCompleted(id, _) => println!("[Plugin] Tool execution completed: {}", id),
            AgentEvent::ToolExecutionFailed(id, err) => println!("[Plugin] Tool execution failed: {} - {}", id, err),
            AgentEvent::LoopFinished => println!("[Plugin] Agent loop finished"),
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session_id, model, message } => {
            println!("Initializing Thunderbot CLI...");

            let config = Config::load();
            let final_model = model.or(config.default_model).unwrap_or_else(|| {
                println!("No model specified and no default found. Run 'thunderbot onboard' to set a default.");
                "gpt-4o".to_string()
            });

            // Use the real routing registry, requires OPENAI_API_KEY or GEMINI_API_KEY
            let routing_registry = RoutingRegistry::new();

            let mut tool_registry = SimpleToolRegistry::new();
            tool_registry.register(Box::new(ReadFileTool));
            tool_registry.register(Box::new(WriteFileTool));
            tool_registry.register(Box::new(ListFilesTool));
            tool_registry.register(Box::new(BashTool { allow_dangerous_commands: false }));

            let session_manager = SimpleSessionManager::new();

            let mut agent_loop = AgentLoop::new(&routing_registry, &tool_registry, &session_manager);
            agent_loop.register_plugin(Box::new(LoggingPlugin));

            println!("Running agent loop for session: {}", session_id);

            if let Some(msg) = message {
                let initial_message = Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    parent_id: None,
                    role: Role::User,
                    content: msg,
                    tool_calls: None,
                };

                agent_loop.run(&session_id, &final_model, vec![initial_message]).await?;

                let session_history = session_manager.load_session(&session_id).await?;
                if let Some(last_message) = session_history.last() {
                    println!("Agent Response: {}", last_message.content);
                }
            } else {
                use std::io::{self, Write};
                println!("Starting REPL mode. Type 'exit' or 'quit' to quit.");
                loop {
                    print!("> ");
                    io::stdout().flush()?;

                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    let input = input.trim();

                    if input == "exit" || input == "quit" {
                        break;
                    }

                    if input.is_empty() {
                        continue;
                    }

                    // For the REPL, we'll want to pass the entire session history to the agent loop.
                    let mut current_history = session_manager.load_session(&session_id).await?;

                    let user_message = Message {
                        id: uuid::Uuid::new_v4().to_string(),
                        parent_id: current_history.last().map(|m| m.id.clone()),
                        role: Role::User,
                        content: input.to_string(),
                        tool_calls: None,
                    };

                    // Add user message to session manager directly so it's recorded
                    session_manager.append_message(&session_id, &user_message).await?;
                    current_history.push(user_message.clone());

                    // The agent loop currently doesn't fetch history itself, it just takes initial_messages
                    // Let's pass the full history to the run method to provide context
                    agent_loop.run(&session_id, &final_model, current_history).await?;

                    let updated_history = session_manager.load_session(&session_id).await?;
                    if let Some(last_message) = updated_history.last() {
                        println!("Agent: {}", last_message.content);
                    }
                }
            }
        }
        Commands::Onboard => {
            let providers = vec!["OpenAI", "Gemini", "Anthropic", "Local/Ollama", "Other"];
            let provider = inquire::Select::new("Choose an LLM provider:", providers).prompt()?;

            let default_model = match provider {
                "OpenAI" => {
                    let models = vec!["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"];
                    inquire::Select::new("Choose a default model:", models).prompt()?.to_string()
                }
                "Gemini" => {
                    let models = vec!["gemini-1.5-pro", "gemini-1.5-flash"];
                    inquire::Select::new("Choose a default model:", models).prompt()?.to_string()
                }
                "Anthropic" => {
                    let models = vec!["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"];
                    inquire::Select::new("Choose a default model:", models).prompt()?.to_string()
                }
                "Local/Ollama" => {
                    inquire::Text::new("Enter the default model name (e.g., llama3):").prompt()?
                }
                _ => {
                    inquire::Text::new("Enter the default model name:").prompt()?
                }
            };

            let mut config = Config::load();
            config.default_model = Some(default_model.clone());
            config.save()?;

            println!("Saved default model '{}' to configuration.", default_model);

            match provider {
                "OpenAI" => {
                    println!("\nMake sure to set the OPENAI_API_KEY environment variable.");
                    println!("You can also create a .env file in your working directory with OPENAI_API_KEY=your_key");
                }
                "Gemini" => {
                    println!("\nMake sure to set the GEMINI_API_KEY environment variable.");
                    println!("You can also create a .env file in your working directory with GEMINI_API_KEY=your_key");
                }
                "Anthropic" => {
                    println!("\nMake sure to set the ANTHROPIC_API_KEY environment variable.");
                    println!("You can also create a .env file in your working directory with ANTHROPIC_API_KEY=your_key");
                }
                "Local/Ollama" => {
                    println!("\nIf using Ollama, you may need to set OPENAI_BASE_URL=http://localhost:11434/v1");
                    println!("and OPENAI_API_KEY=ollama (or any dummy value).");
                }
                _ => {}
            }
        }
        Commands::RunTool { tool, args } => {
            println!("Running tool: {}", tool);
            let mut tool_registry = SimpleToolRegistry::new();
            tool_registry.register(Box::new(ReadFileTool));
            tool_registry.register(Box::new(WriteFileTool));
            tool_registry.register(Box::new(ListFilesTool));
            tool_registry.register(Box::new(BashTool { allow_dangerous_commands: false }));

            let parsed_args: Value = serde_json::from_str(&args)?;
            match tool_registry.execute_tool(&tool, parsed_args).await {
                Ok(result) => println!("Result:\n{}", serde_json::to_string_pretty(&result)?),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
    }

    Ok(())
}
