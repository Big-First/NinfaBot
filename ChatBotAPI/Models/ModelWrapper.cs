using System.Text.Json.Serialization;

namespace ChatBotAPI.Models;

/// <summary>
/// Wrapper class that contains model information and version details.
/// </summary>
public class ModelWrapper
{
    /// <summary>
    /// Gets or sets the version of the model.
    /// </summary>
    public string Version { get; set; }

    /// <summary>
    /// Gets or sets the model configuration and details.
    /// </summary>
    public Model Model { get; set; }
}