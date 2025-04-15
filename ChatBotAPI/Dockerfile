FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["ChatBotAPI/ChatBotAPI.csproj", "ChatBotAPI/"]
RUN dotnet restore "ChatBotAPI/ChatBotAPI.csproj"
COPY . .
WORKDIR "/src/ChatBotAPI"
RUN dotnet build "ChatBotAPI.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ChatBotAPI.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ChatBotAPI.dll"]
