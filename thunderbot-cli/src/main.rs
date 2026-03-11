use anyhow::Result;
use clap::{Parser, Subcommand};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;

use thunderbot_core::{
    AgentEvent, AgentLoop, CompletionResponse, Message, ModelRegistry, Plugin, Role, SessionManager, StopReason, Tool, ToolDefinition, ToolRegistry
};
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

        /// The model to use
        #[arg(short, long, default_value = "dummy-model")]
        model: String,

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
}

// Simple Model Registry
struct SimpleModelRegistry;

#[async_trait]
impl ModelRegistry for SimpleModelRegistry {
    async fn generate_completion(
        &self,
        _model: &str,
        messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<CompletionResponse> {
        // Just echo back the last message for now
        let last_msg = messages.last().map(|m| m.content.clone()).unwrap_or_default();

        let msg = Message {
            id: uuid::Uuid::new_v4().to_string(),
            parent_id: messages.last().map(|m| m.id.clone()),
            role: Role::Assistant,
            content: format!("Echo: {}", last_msg),
            tool_calls: None,
        };

        Ok(CompletionResponse {
            message: msg,
            stop_reason: StopReason::EndTurn,
        })
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
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session_id, model, message } => {
            println!("Initializing Thunderbot CLI...");

            let model_registry = SimpleModelRegistry;

            let mut tool_registry = SimpleToolRegistry::new();
            tool_registry.register(Box::new(ReadFileTool));
            tool_registry.register(Box::new(WriteFileTool));
            tool_registry.register(Box::new(ListFilesTool));
            tool_registry.register(Box::new(BashTool { allow_dangerous_commands: false }));

            let session_manager = SimpleSessionManager::new();

            let mut agent_loop = AgentLoop::new(&model_registry, &tool_registry, &session_manager);
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

                agent_loop.run(&session_id, &model, vec![initial_message]).await?;

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
                    agent_loop.run(&session_id, &model, current_history).await?;

                    let updated_history = session_manager.load_session(&session_id).await?;
                    if let Some(last_message) = updated_history.last() {
                        println!("Agent: {}", last_message.content);
                    }
                }
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
