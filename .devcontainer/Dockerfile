FROM mcr.microsoft.com/devcontainers/dotnet:1-8.0-bookworm

# Optional: Install additional tools if needed
# RUN apt-get update && apt-get install -y git curl

# Install EF CLI for migrations (optional)
USER vscode
RUN dotnet tool install --global dotnet-ef
