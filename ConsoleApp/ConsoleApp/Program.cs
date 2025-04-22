using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using System.Threading.Tasks;
using ConsoleApp;

class Program
{
    static async Task Main(string[] args)
    {
        string inputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
        // Let's use a different output name to avoid confusion
        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer_bpe.json");
        
        int totalTuples = 1500;
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys","ChatbotTrainingDataWithExpandedCategories.json");

        var inputs = new[]
        {
            "Hey", "Hi", "What's up?", "How’s it going?", "Tell me about yourself", 
            "What do you do for a living?", "Do you like traveling?", "How was your weekend?", 
            "You have the most beautiful smile!", "I’m really bad at picking outfits.",
            "If I were a superhero, my power would be... to make you laugh.", 
            "What’s something you’re passionate about?", "Do you believe in soulmates?", 
            "What do you think love is?", "Are you a morning person or a night owl?",
            "Any fun plans for the weekend?", "What’s your idea of a perfect date?", 
            "Do you like pets?", "What’s your favorite movie?", "What’s your favorite hobby?"
        };

        var responses = new[]
        {
            "Hi there! How can I help you today?", "Thanks! That means a lot.", "I'm here looking to meet someone cool.",
            "I enjoy hiking, reading, and trying new foods.", "That’s sweet of you to say.",
            "I’m open to something serious if it feels right.", "I do! I have a dog named Max.", 
            "I’m more of a movie person, but I enjoy a good book too.", "Let’s plan something soon.",
            "I live downtown, you?", "I love learning about people. What about you?",
            "Honestly just curious to see who’s out here.", "I work in tech. You?", 
            "I like calm weekends and spontaneous plans.", "Tell me something fun about you.", 
            "Do you believe in love at first match?", "I’m usually pretty chill, but I love deep talks.",
            "Sounds like we might get along.", "What’s your favorite way to spend a Sunday?",
            "Aww, thank you! Your smile isn’t so bad either", "Well, I think you’d look great in anything.",
            "Haha, I think I’d love that power. Laughter is the best thing in the world!", 
            "I’m passionate about learning new things and helping others. How about you?", 
            "I believe love is about understanding and growing together. What do you think?", 
            "Love is about connection, trust, and being there for each other.",
            "I think there’s someone out there for everyone, but it’s also about making it work. What do you think?",
            "I love photography. It helps me see the world in a different way. What about you?", 
            "I’m really into soccer. How about you?", "Italy is amazing! The food, culture, and people are all so great.",
            "I’d love to visit Japan. The culture is fascinating.", "I’m a huge fan of fantasy books. What about you?", 
            "Yes, I have a cat named Oliver. How about you?", "I think I’d love to have a pet fox. They’re so cute.",
            "Sushi is my favorite food. It’s so fresh and delicious!", "I prefer coffee. I need it to start my day.",
            "Thank you! I’ve always liked them too", "That’s so sweet of you to say. I try my best to be kind."
        };

        var categories = new[] { "flirting", "smalltalk", "deep", "hobbies & interests", "travel", "movies & books", "pets & animals", "food & drink", "compliments", "first date ideas" };
        var options = new JsonSerializerOptions { WriteIndented = false };
        var random = new Random();

        using (var writer = new StreamWriter(filePath))
        {
            for (int i = 0; i < totalTuples; i++)
            {
                // Randomly select input, response, and category
                var input = inputs[random.Next(inputs.Length)];
                var output = responses[random.Next(responses.Length)] + "<|endoftext|>";

                var pair = new
                {
                    input = input,
                    output = output
                };

                string jsonLine = JsonSerializer.Serialize(pair, options);
                writer.WriteLine(jsonLine);

                // Status update for large datasets
                if ((i + 1) % 1500 == 0)
                    Console.WriteLine($"{i + 1} tuplas geradas...");
            }
        }

        Console.WriteLine($"Arquivo gerado com sucesso: {filePath}");
    }
}