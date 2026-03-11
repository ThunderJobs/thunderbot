use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a message in the conversation tree.
///
/// Features `parent_id` to enable native branching/forking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: String,
    // Add additional metadata if needed (e.g., tool calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub message: Message,
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Abstract LLM Provider trait to route requests to multiple providers.
#[async_trait]
pub trait ModelRegistry: Send + Sync {
    async fn generate_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<CompletionResponse>;
}

/// A definition of a Tool passed to the Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value, // JSON schema
}

/// Trait to define a Tool's executable behavior.
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, args: Value) -> anyhow::Result<Value>;
}

/// Manages the registration and dispatching of tools.
#[async_trait]
pub trait ToolRegistry: Send + Sync {
    fn register(&mut self, tool: Box<dyn Tool>);
    async fn execute_tool(&self, name: &str, args: Value) -> anyhow::Result<Value>;
    fn definitions(&self) -> Vec<ToolDefinition>;
}

/// Manages the context tree, including JSONL tree persistence and compaction.
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Loads the message tree from the persistence layer.
    async fn load_session(&self, session_id: &str) -> anyhow::Result<Vec<Message>>;
    
    /// Appends a new message to the persistence layer.
    async fn append_message(&self, session_id: &str, message: &Message) -> anyhow::Result<()>;
    
    /// Triggers context compaction (truncate, summarize, compact).
    async fn compact_context(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Event types for the plugin system
#[derive(Debug, Clone)]
pub enum AgentEvent {
    LoopStarted,
    CompletionGenerated(CompletionResponse),
    ToolExecutionStarted(ToolCall),
    ToolExecutionCompleted(String, Value),
    ToolExecutionFailed(String, String),
    LoopFinished,
}

/// Trait to define an Event Plugin
#[async_trait]
pub trait Plugin: Send + Sync {
    async fn on_event(&self, event: &AgentEvent) -> anyhow::Result<()>;
}

/// The core loop chassis that drives the agent interaction.
pub struct AgentLoop<'a> {
    pub model_registry: &'a dyn ModelRegistry,
    pub tool_registry: &'a dyn ToolRegistry,
    pub session_manager: &'a dyn SessionManager,
    pub plugins: Vec<Box<dyn Plugin + 'a>>,
}

impl<'a> AgentLoop<'a> {
    pub fn new(
        model_registry: &'a dyn ModelRegistry,
        tool_registry: &'a dyn ToolRegistry,
        session_manager: &'a dyn SessionManager,
    ) -> Self {
        Self {
            model_registry,
            tool_registry,
            session_manager,
            plugins: Vec::new(),
        }
    }

    pub fn register_plugin(&mut self, plugin: Box<dyn Plugin + 'a>) {
        self.plugins.push(plugin);
    }
    
    async fn dispatch_event(&self, event: AgentEvent) -> anyhow::Result<()> {
         for plugin in &self.plugins {
             plugin.on_event(&event).await?;
         }
         Ok(())
    }

    /// The fundamental agent loop
    pub async fn run(&self, session_id: &str, model: &str, initial_messages: Vec<Message>) -> anyhow::Result<()> {
        let mut messages = initial_messages;
        
        self.dispatch_event(AgentEvent::LoopStarted).await?;

        loop {
            // Check context bounds and compact if necessary
            self.session_manager.compact_context(session_id).await?;
            
            let tools = self.tool_registry.definitions();
            
            let response = self.model_registry.generate_completion(model, &messages, &tools).await?;
            self.dispatch_event(AgentEvent::CompletionGenerated(response.clone())).await?;
            self.session_manager.append_message(session_id, &response.message).await?;
            messages.push(response.message.clone());
            
            if response.stop_reason != StopReason::ToolUse {
                break;
            }
            
            if let Some(tool_calls) = response.message.tool_calls {
                for tool_call in tool_calls {
                     self.dispatch_event(AgentEvent::ToolExecutionStarted(tool_call.clone())).await?;
                     match self.tool_registry.execute_tool(&tool_call.name, tool_call.arguments.clone()).await {
                         Ok(result) => {
                             self.dispatch_event(AgentEvent::ToolExecutionCompleted(tool_call.id.clone(), result.clone())).await?;
                             let tool_message = Message { 
                                 id: uuid::Uuid::new_v4().to_string(), // In a real system, you'd use a better ID generator
                                 parent_id: Some(response.message.id.clone()),
                                 role: Role::Tool, 
                                 content: result.to_string(),
                                 tool_calls: None
                             };
                             messages.push(tool_message.clone());
                             self.session_manager.append_message(session_id, &tool_message).await?;
                         },
                         Err(e) => {
                             self.dispatch_event(AgentEvent::ToolExecutionFailed(tool_call.id.clone(), e.to_string())).await?;
                             let tool_error_message = Message {
                                 id: uuid::Uuid::new_v4().to_string(), // In a real system, you'd use a better ID generator
                                 parent_id: Some(response.message.id.clone()),
                                 role: Role::Tool,
                                 content: format!("Error executing tool: {}", e),
                                 tool_calls: None
                             };
                             messages.push(tool_error_message.clone());
                             self.session_manager.append_message(session_id, &tool_error_message).await?;
                         }
                     }
                }
            } else {
                 break; // Model indicated tool use but provided no tools
            }
        }
        
        self.dispatch_event(AgentEvent::LoopFinished).await?;
        
        Ok(())
    }
}
