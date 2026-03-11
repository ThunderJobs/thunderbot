use thunderbot_core::{Tool, ToolDefinition};
use async_trait::async_trait;
use serde_json::Value;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Reads the contents of a file.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, args: Value) -> anyhow::Result<Value> {
        let path = args["path"].as_str().ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let content = tokio::fs::read_to_string(path).await?;
        Ok(serde_json::Value::String(content))
    }
}

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Writes content to a file.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(&self, args: Value) -> anyhow::Result<Value> {
        let path = args["path"].as_str().ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let content = args["content"].as_str().ok_or_else(|| anyhow::anyhow!("Missing content"))?;
        tokio::fs::write(path, content).await?;
        Ok(serde_json::Value::String("Success".to_string()))
    }
}

pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_files".to_string(),
            description: "Lists files in a directory.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list files from"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, args: Value) -> anyhow::Result<Value> {
        let path = args["path"].as_str().ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let mut entries = tokio::fs::read_dir(path).await?;
        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            if let Ok(file_name) = entry.file_name().into_string() {
                files.push(file_name);
            }
        }
        Ok(serde_json::json!(files))
    }
}

pub struct BashTool {
    pub allow_dangerous_commands: bool,
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Executes a bash command.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, args: Value) -> anyhow::Result<Value> {
        let command = args["command"].as_str().ok_or_else(|| anyhow::anyhow!("Missing command"))?;
        
        if !self.allow_dangerous_commands {
            if command.contains("rm ") || command.contains("mv ") || command.contains("wget ") || command.contains("curl ") {
                 return Err(anyhow::anyhow!("Command is considered dangerous and is safely gated."));
            }
        }

        let output = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(serde_json::json!({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": output.status.code()
        }))
    }
}
