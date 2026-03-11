use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;

use thunderbot_core::{
    CompletionResponse, Message, ModelRegistry, Role, StopReason, ToolCall, ToolDefinition,
};

/// Models registry that delegates to sub-registries based on prefix matching
pub struct RoutingRegistry {
    openai_client: Option<OpenAIClient>,
    gemini_client: Option<GeminiClient>,
}

impl RoutingRegistry {
    pub fn new() -> Self {
        // Initialize clients if environment variables are present
        let openai_client = env::var("OPENAI_API_KEY").ok().map(|key| {
            let base_url = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            OpenAIClient::new(key, base_url)
        });

        let gemini_client = env::var("GEMINI_API_KEY").ok().map(GeminiClient::new);

        Self {
            openai_client,
            gemini_client,
        }
    }
}

#[async_trait]
impl ModelRegistry for RoutingRegistry {
    async fn generate_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<CompletionResponse> {
        let lower_model = model.to_lowercase();

        if lower_model.starts_with("gemini") {
            if let Some(client) = &self.gemini_client {
                return client.generate_completion(model, messages, tools).await;
            } else {
                anyhow::bail!("Model {} requested but GEMINI_API_KEY is not set", model);
            }
        }

        // Assume everything else uses the OpenAI protocol (including qwen on OpenAI compatible endpoints)
        if let Some(client) = &self.openai_client {
            return client.generate_completion(model, messages, tools).await;
        } else {
            anyhow::bail!("Model {} requested but OPENAI_API_KEY is not set", model);
        }
    }
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionDefinition,
}

#[derive(Serialize)]
struct OpenAIFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl ModelRegistry for OpenAIClient {
    async fn generate_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<CompletionResponse> {
        let mut req_messages = Vec::new();

        for msg in messages {
            let role_str = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };

            // If the message is a tool result, we need to map the ID
            // Here we do a somewhat hacky approach where the tool call ID is inferred from the parent ID,
            // or we just use a dummy if not found since the core doesn't strictly track tool_call_id in the message yet.
            // Assuming `parent_id` is the `tool_call_id` for simplicity if role is Tool.
            let tool_call_id = if msg.role == Role::Tool {
                Some(msg.parent_id.clone().unwrap_or_else(|| "unknown".to_string()))
            } else {
                None
            };

            let tool_calls = msg.tool_calls.as_ref().map(|calls| {
                calls.iter().map(|tc| OpenAIToolCall {
                    id: tc.id.clone(),
                    call_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                }).collect()
            });

            req_messages.push(OpenAIMessage {
                role: role_str.to_string(),
                content: if msg.role == Role::Tool && msg.content.is_empty() { None } else { Some(msg.content.clone()) },
                tool_calls,
                tool_call_id,
            });
        }

        let mut body = json!({
            "model": model,
            "messages": req_messages,
        });

        if !tools.is_empty() {
            let oa_tools: Vec<OpenAITool> = tools.iter().map(|t| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunctionDefinition {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            }).collect();
            body.as_object_mut().unwrap().insert("tools".to_string(), json!(oa_tools));
        }

        let url = format!("{}/chat/completions", self.base_url);
        let res = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        let status = res.status();
        let text = res.text().await?;

        if !status.is_success() {
            anyhow::bail!("OpenAI API error ({}): {}", status, text);
        }

        let res_json: Value = serde_json::from_str(&text)?;

        let choice = &res_json["choices"][0];
        let message = &choice["message"];
        let finish_reason = choice["finish_reason"].as_str().unwrap_or("");

        let stop_reason = match finish_reason {
            "tool_calls" | "function_call" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let content = message["content"].as_str().unwrap_or("").to_string();

        let mut parsed_tool_calls = None;
        if let Some(tcs) = message["tool_calls"].as_array() {
            let mut calls = Vec::new();
            for tc in tcs {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let function = &tc["function"];
                let name = function["name"].as_str().unwrap_or("").to_string();
                let arguments_str = function["arguments"].as_str().unwrap_or("{}");
                let arguments = serde_json::from_str(arguments_str).unwrap_or_else(|_| json!({}));

                calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            if !calls.is_empty() {
                parsed_tool_calls = Some(calls);
            }
        }

        let id = res_json["id"].as_str().unwrap_or_else(|| "").to_string();

        let core_msg = Message {
            id,
            parent_id: messages.last().map(|m| m.id.clone()),
            role: Role::Assistant,
            content,
            tool_calls: parsed_tool_calls,
        };

        Ok(CompletionResponse {
            message: core_msg,
            stop_reason,
        })
    }
}

pub struct GeminiClient {
    client: Client,
    api_key: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    #[serde(rename_all = "camelCase")]
    FunctionCall { function_call: GeminiFunctionCall },
    #[serde(rename_all = "camelCase")]
    FunctionResponse { function_response: GeminiFunctionResponse },
}

#[derive(Serialize, Deserialize, Clone)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl ModelRegistry for GeminiClient {
    async fn generate_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<CompletionResponse> {
        let mut contents = Vec::new();

        for msg in messages {
            // Gemini uses "user" and "model" roles
            let role_str = match msg.role {
                Role::System | Role::User => "user",
                Role::Assistant => "model",
                Role::Tool => "user", // Gemini sends function responses as user
            };

            let mut parts = Vec::new();

            if msg.role == Role::Tool {
                // Tool response
                // Gemini functionResponse format
                // In our core structure, `msg.content` has the JSON result. We need the function name.
                // We'll hackily extract it if possible, or just default to "tool"
                // To properly map, we need the tool name. We might have to guess or store it.
                // Let's assume the content is a JSON with some structure or we just wrap it.
                let val: Value = serde_json::from_str(&msg.content).unwrap_or_else(|_| json!({"result": msg.content}));
                parts.push(GeminiPart::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name: "tool_response".to_string(), // This is a limitation, we should know the original tool name
                        response: val,
                    }
                });
            } else if !msg.content.is_empty() {
                parts.push(GeminiPart::Text { text: msg.content.clone() });
            }

            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    parts.push(GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: tc.name.clone(),
                            args: tc.arguments.clone(),
                        }
                    });
                }
            }

            if !parts.is_empty() {
                contents.push(GeminiContent {
                    role: role_str.to_string(),
                    parts,
                });
            }
        }

        let mut body = json!({
            "contents": contents,
        });

        if !tools.is_empty() {
            let decls: Vec<GeminiFunctionDeclaration> = tools.iter().map(|t| GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            }).collect();

            body.as_object_mut().unwrap().insert("tools".to_string(), json!([
                { "functionDeclarations": decls }
            ]));
        }

        // e.g. gemini-1.5-pro
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, self.api_key
        );

        let res = self.client.post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        let status = res.status();
        let text = res.text().await?;

        if !status.is_success() {
            anyhow::bail!("Gemini API error ({}): {}", status, text);
        }

        let res_json: Value = serde_json::from_str(&text)?;

        let candidate = &res_json["candidates"][0];
        let content_parts = candidate["content"]["parts"].as_array().unwrap();

        let mut content_str = String::new();
        let mut tool_calls = Vec::new();

        for part in content_parts {
            if let Some(text) = part["text"].as_str() {
                content_str.push_str(text);
            } else if let Some(func_call) = part.get("functionCall") {
                let name = func_call["name"].as_str().unwrap_or("").to_string();
                let args = func_call["args"].clone();
                tool_calls.push(ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    name,
                    arguments: args,
                });
            }
        }

        let stop_reason = if !tool_calls.is_empty() {
            StopReason::ToolUse
        } else {
            let finish_reason = candidate["finishReason"].as_str().unwrap_or("");
            if finish_reason == "MAX_TOKENS" {
                StopReason::MaxTokens
            } else {
                StopReason::EndTurn
            }
        };

        let core_msg = Message {
            id: uuid::Uuid::new_v4().to_string(),
            parent_id: messages.last().map(|m| m.id.clone()),
            role: Role::Assistant,
            content: content_str,
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        };

        Ok(CompletionResponse {
            message: core_msg,
            stop_reason,
        })
    }
}
