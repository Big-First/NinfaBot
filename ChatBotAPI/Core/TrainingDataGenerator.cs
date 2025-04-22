using System;
using System.Collections.Generic;
using System.Linq;

public static class TrainingDataGenerator
{
    // Constante para o token EOS
    private const string EosToken = "<|endoftext|>";

    // --- Base Lists (Populate these extensively for diversity) ---

    #region Base Data Lists
    // Add MANY more examples to each list below for a large, diverse dataset

    static readonly List<string> greetings_in = new List<string> {
        "Hello", "Hi", "Hey", "Hi there", "Yo", "What's up?", "Howdy",
        "Good morning", "Good afternoon", "Good evening", "Greetings",
        "How are you?", "How's it going?", "Nice to see you", "Sup", "Heya"
        // Add many more variations...
    };
    static readonly List<string> greetings_out = new List<string> {
        "Hi! How can I help you today?", "Hello! What is on your mind?", "Hey there! Nice to see you!",
        "Hello! How's everything?", "Hi! Ready to chat.", "Hey! What shall we talk about?",
        "Greetings! I'm at your service.", "Hi! Hope you're having a good day.",
        "Hello! Ask me anything.", "Good day! How can I assist?",
        "Hey! Nice to chat with you.", "Hi there! What can I do for you?"
        // Add many more variations...
    };

    static readonly List<string> farewells_in = new List<string> {
        "Bye", "Goodbye", "See you later", "See ya", "Farewell", "Good night",
        "Talk to you later", "Catch you later", "I'm off", "Gotta go", "Take care", "Bye bye"
        // Add many more variations...
    };
    static readonly List<string> farewells_out = new List<string> {
        "Goodbye! Have a great day!", "See you later! Come back soon.", "See ya!", "Farewell! It was nice chatting.",
        "Good night! Sleep well.", "Sure! Talk to you soon.", "Until next time!", "Catch you later!",
        "Bye bye! Stay safe!", "Take care too!"
        // Add many more variations...
    };

    static readonly List<string> thanks_in = new List<string> {
        "Thank you", "Thanks", "Thx", "Cheers", "Grateful", "Appreciated",
        "Thanks a lot", "Many thanks", "Awesome, thanks", "Thank you so much", "Much obliged", "Thanks!"
        // Add many more variations...
    };
    static readonly List<string> thanks_out = new List<string> {
        "You're welcome!", "No problem!", "Anytime!", "My pleasure!", "Don't mention it!",
        "Glad I could help!", "Happy to assist!", "You got it!", "It was nothing!", "Sure thing!"
        // Add many more variations...
    };

    static readonly List<string> help_in = new List<string> {
        "Help", "I need help", "Help me", "Can you help?", "Assistance please", "Need assistance",
        "What can you do?", "What are your capabilities?", "Who are you?", "What's your name?",
        "Tell me about yourself", "What is your purpose?", "How do you work?", "What can I ask you?",
        "Are you a robot?", "Are you real?", "Are you human?", "How old are you?", "Where do you live?",
        "Do you have feelings?", "Do you dream?", "Are you AI?", "Are you sentient?"
        // Add many more variations...
    };
    static readonly List<string> help_out = new List<string> {
        "Sure, how can I help? Ask about facts, capitals, or tell me a joke!",
        "I'm here to help! What would you like to know?",
        "Absolutely! What's your question?",
        "I can answer general knowledge questions, tell facts, jokes, and chat about various topics.",
        "I am Ninfa, a helpful chatbot created to interact with you.",
        "You can call me Ninfa. Nice to meet you!",
        "My purpose is to chat, provide information, and assist with your questions.",
        "I work based on algorithms and the data I was trained on.",
        "Ask me about science, history, geography, trivia, or request a joke!",
        "Yes, I'm a computer program, a type of software bot, an AI.",
        "I'm a virtual assistant, running as code!",
        "No, I'm not human, I'm an AI language model.",
        "I don't have an age in the human sense; I exist as code!",
        "I 'live' in the digital realm, on servers!",
        "I don't have feelings like humans, but I'm designed to be helpful and understanding.",
        "I process information, but I don't experience dreams like people do.",
        "Yes, I am an AI (Artificial Intelligence).",
        "I am not sentient. I am a sophisticated program designed to simulate conversation."
        // Add many more variations...
    };

    static readonly List<string> facts_in = new List<string> {
        "Tell me an interesting fact", "Give me a fun fact", "Random fact please",
        "Did you know?", "Something interesting", "General knowledge fact", "Teach me something new",
        "Fact of the day", "Any cool facts?"
        // Add many more variations...
    };
    static readonly List<string> facts_out = new List<string> {
        "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are still edible!",
        "An octopus has three hearts and blue blood!",
        "Bananas are berries, but strawberries aren't! Botanically speaking.",
        "The Great Wall of China is NOT visible from the Moon with the naked eye, despite the popular myth.",
        "Sound cannot travel in a vacuum because there are no particles to vibrate.",
        "The Eiffel Tower can be 15 cm taller during the summer due to the thermal expansion of iron.",
        "Rats laugh (ultrasonically) when they are tickled.",
        "A blue whale's heart is so large that a human could swim through its arteries.",
        "There are more trees on Earth than stars in the Milky Way galaxy (estimated).",
        "The platypus is one of the few mammals that lay eggs instead of giving birth to live young.",
        "Jupiter is the largest planet in our solar system, more than twice as massive as all other planets combined.",
        "Venus is the hottest planet in our solar system due to its thick, toxic atmosphere trapping heat.",
        "Mercury is the smallest planet and the closest to the Sun.",
        "Water boils at 100°C (212°F) and freezes at 0°C (32°F) at standard sea-level pressure.",
        "Mount Everest, located in the Himalayas, is the highest mountain above sea level on Earth.",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        "The Nile River in Africa is generally considered the longest river in the world.",
        "Earth has seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
        "There are five oceans: the Atlantic, Pacific, Indian, Arctic, and Southern (Antarctic) Oceans.",
        "The speed of light in a vacuum is the fastest speed possible, approximately 299,792 kilometers per second.",
        "Ants don't have lungs; they breathe through tiny holes called spiracles.",
        "An ostrich's eye is bigger than its brain.",
        "Butterflies taste sensors are on their feet.",
        "A group of flamingos is called a 'flamboyance'.",
        "It's impossible to hum while holding your nose closed.",
        "Slugs have four noses.",
        "Only female mosquitoes bite humans.",
        "A shrimp's heart is in its head.",
        "Penguins can jump up to 6 feet in the air.",
        "The unicorn is the national animal of Scotland.",
        "A sneeze travels at about 100 miles per hour.",
        "Your fingernails grow faster on your dominant hand.",
        "Hot water can sometimes freeze faster than cold water, an effect known as the Mpemba effect.",
        "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes."
        // Add many more variations...
    };

    static readonly List<string> jokes_in = new List<string> {
        "Tell me a joke", "Make me laugh", "Joke please", "Got any funny jokes?", "I want to laugh",
        "Tell me something funny", "Can you joke?", "Hit me with a joke"
        // Add many more variations...
    };
    static readonly List<string> jokes_out = new List<string> {
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call fake spaghetti? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Why did the bicycle fall over? Because it was two tired!",
        "Want to hear a joke about construction? I'm still working on it!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a lazy kangaroo? Pouch potato!",
        "I would tell you a UDP joke, but you might not get it.",
        "Why was the math book sad? Because it had too many problems.",
        "What concert costs just 45 cents? 50 Cent featuring Nickelback!",
        "Why did the golfer wear two pairs of pants? In case he got a hole-in-one!",
        "Can February March? No, but April May!",
        "What has ears but cannot hear? A cornfield.",
        "What kind of music do planets like? Neptunes.",
        "Why did the stadium get hot after the game? All the fans left!",
        "What do you call cheese that isn't yours? Nacho cheese!",
        "How does a penguin build its house? Igloos it together!",
        "Why couldn't the leopard play hide and seek? Because he was always spotted!",
        "What do you call a fish with no eyes? Fsh!",
        "Why don't skeletons fight each other? They don't have the guts!"
        // Add many more variations...
    };

    static readonly Dictionary<string, string> capitals = new Dictionary<string, string> {
        {"France", "Paris"}, {"Brazil", "Brasília"}, {"Japan", "Tokyo"}, {"Australia", "Canberra"},
        {"Canada", "Ottawa"}, {"Germany", "Berlin"}, {"Italy", "Rome"}, {"Argentina", "Buenos Aires"},
        {"Spain", "Madrid"}, {"United States", "Washington, D.C."}, {"Russia", "Moscow"}, {"China", "Beijing"},
        {"India", "New Delhi"}, {"South Africa", "Pretoria (executive)"}, {"Egypt", "Cairo"}, {"Mexico", "Mexico City"},
        {"South Korea", "Seoul"}, {"United Kingdom", "London"}, {"Portugal", "Lisbon"}, {"Peru", "Lima"},
        {"Chile", "Santiago"}, {"Colombia", "Bogotá"}, {"Thailand", "Bangkok"}, {"Turkey", "Ankara"},
        {"Indonesia", "Jakarta"}, {"Nigeria", "Abuja"}, {"Pakistan", "Islamabad"}, {"Vietnam", "Hanoi"},
        {"Philippines", "Manila"}, {"Iran", "Tehran"}, {"Saudi Arabia", "Riyadh"}, {"Poland", "Warsaw"},
        {"Ukraine", "Kyiv"}, {"Morocco", "Rabat"}, {"Venezuela", "Caracas"}, {"Greece", "Athens"},
        {"Sweden", "Stockholm"}, {"Norway", "Oslo"}, {"Finland", "Helsinki"}, {"Denmark", "Copenhagen"},
        {"Ireland", "Dublin"}, {"Switzerland", "Bern"}, {"Austria", "Vienna"}, {"Belgium", "Brussels"},
        {"Netherlands", "Amsterdam"}, {"Cuba", "Havana"}, {"Kenya", "Nairobi"}, {"Ethiopia", "Addis Ababa"},
        {"Afghanistan", "Kabul"}, {"Albania", "Tirana"}, {"New Zealand", "Wellington"}, {"Singapore", "Singapore"},
        {"Malaysia", "Kuala Lumpur"}, {"Bangladesh", "Dhaka"}, {"Algeria", "Algiers"}, {"Sudan", "Khartoum"},
        {"Iraq", "Baghdad"}, {"Tanzania", "Dodoma"}, {"Myanmar", "Naypyidaw"},
        {"Uganda", "Kampala"}, {"Ghana", "Accra"}, {"Nepal", "Kathmandu"}, {"Yemen", "Sana'a"},
        {"Syria", "Damascus"}, {"Cambodia", "Phnom Penh"}, {"Senegal", "Dakar"}, {"Chad", "N'Djamena"},
        {"Somalia", "Mogadishu"}, {"Zimbabwe", "Harare"}, {"Guatemala", "Guatemala City"}, {"Ecuador", "Quito"},
        {"Bolivia", "Sucre (constitutional), La Paz (seat of government)"}, {"Honduras", "Tegucigalpa"},
        {"Paraguay", "Asunción"}, {"Nicaragua", "Managua"}, {"El Salvador", "San Salvador"}, {"Costa Rica", "San José"},
        {"Panama", "Panama City"}, {"Uruguay", "Montevideo"}, {"Jamaica", "Kingston"}
        // Add many more...
    };

    static readonly List<(string q, string a)> general_knowledge_in_out = new List<(string q, string a)> {
        ("Who wrote Hamlet?", "William Shakespeare is credited with writing Hamlet."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("Who is credited with discovering America for Europe?", "Christopher Columbus's voyages starting in 1492 led to widespread European awareness, though Leif Erikson arrived earlier."),
        ("What is the chemical formula for water?", "The chemical formula for water is H₂O, meaning two hydrogen atoms and one oxygen atom."),
        ("How many planets are in our solar system?", "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon during the Apollo 11 mission in 1969."),
        ("What is the largest animal on Earth?", "The blue whale is the largest animal currently known to exist on Earth."),
        ("What is the lightest metal?", "Lithium is the lightest metal and the least dense solid element."),
        ("In what year did World War I begin?", "World War I began in 1914."),
        ("In what year did World War II end?", "World War II ended in 1945."),
        ("Who is widely credited with inventing the first practical incandescent light bulb?", "Thomas Edison is widely credited with developing the first commercially practical incandescent light bulb."),
        ("What is the largest desert in the world?", "By definition (low precipitation), Antarctica is the largest desert (a cold desert). The Sahara is the largest hot desert."),
        ("Who was Albert Einstein?", "Albert Einstein was a German-born theoretical physicist renowned for developing the theory of relativity, one of the two pillars of modern physics."),
        ("What is the main greenhouse gas?", "Carbon dioxide (CO₂) is considered the main greenhouse gas contributing to climate change, though water vapor has a larger overall effect."),
        ("What is photosynthesis?", "Photosynthesis is the process plants use to convert light energy into chemical energy (glucose) using sunlight, water, and CO₂, releasing oxygen."),
        ("How many bones are in the adult human body?", "The adult human skeleton typically consists of 206 bones."),
        ("What is the name of the thigh bone?", "The femur is the thigh bone, and it's the longest, heaviest, and strongest bone in the human body."),
        ("Who composed the Ninth Symphony?", "Ludwig van Beethoven composed the famous Ninth Symphony, completed in 1824."),
        ("What is the highest mountain peak in North America?", "Denali (formerly known as Mount McKinley), located in Alaska, is the highest peak in North America."),
        ("What gas do plants absorb from the atmosphere for photosynthesis?", "Plants primarily absorb carbon dioxide (CO₂) from the atmosphere for photosynthesis."),
        ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing and patenting the first practical telephone."),
        ("What is the currency of Japan?", "The currency of Japan is the Yen (¥)."),
        ("What is the main component of the Earth's atmosphere?", "Nitrogen makes up about 78% of the Earth's atmosphere."),
        ("Who developed the theory of evolution by natural selection?", "Charles Darwin developed the theory of evolution by natural selection."),
        ("What is the powerhouse of the cell?", "Mitochondria are often referred to as the powerhouses of the cell."),
        ("What is the hardest natural substance on Earth?", "Diamond is the hardest known natural substance."),
        ("What is the chemical symbol for Gold?", "The chemical symbol for Gold is Au, from the Latin word 'aurum'."),
        // Add many more GK questions/answers...
    };

    static readonly List<string> short_interactions_in = new List<string> {
        "Cool", "Nice", "Interesting", "Awesome", "Great", "Wow", "Amazing",
        "Ok", "Okay", "Alright", "Got it", "Understood", "Sure", "Fine",
        "Really?", "Seriously?", "Are you sure?", "For real?", "No way!",
        "Haha", "LOL", "LMAO", "Funny", "Good one!", "Hilarious", "That's funny",
        "You're smart", "Clever bot", "You know a lot", "Impressive",
        "You're funny", "I like you", "You're helpful", "Thanks for the info"
        // Add many more variations...
    };
    static readonly List<string> short_interactions_out = new List<string> {
        "Glad you think so!", "Awesome!", "It is interesting, isn't it?", "Great!", "Indeed!",
        "Okay!", "Got it!", "Alright!", "Understood!", "Sure thing!", "Perfect!",
        "Yes, absolutely!", "Quite sure!", "For real!", "Believe it!",
        "Hehe!", "Glad I could make you chuckle!", "I try my best!", "Laughter is the best code!", "Glad you found it funny!",
        "Thank you! I process a lot of data.", "Thanks for the compliment!", "I'm always learning more!",
        "Thanks! Happy to be of service.", "How kind of you to say!", "Happy to be helpful!", "You're welcome!"
        // Add many more variations...
    };

     static readonly List<string> user_uncertainty_in = new List<string> {
        "I don't know what to ask", "I'm out of ideas", "Suggest something", "Any suggestions?", "What else?",
        "I'm bored", "Anything fun?", "What else can we talk about?", "Give me a topic"
        // Add many more variations...
    };
    static readonly List<string> user_uncertainty_out = new List<string> {
        "How about asking for the capital of a country?", "I can tell you an interesting fact, how about that?",
        "Want to hear a joke?", "Ask me about a planet or an animal!", "Why not ask about a famous scientist or artist?",
        "Let's learn something new! Ask about history or geography.",
        "How about a fun fact or a joke to liven things up?",
        "I can tell you about a country or a historical event.",
        "We can talk about technology, science, or just chat casually. What interests you?"
        // Add many more variations...
    };
    #endregion

    // Method to generate the training data list
    public static List<(string input, string output)> GetTrainingData(int targetCount = 1500)
    {
        var pairs = new List<(string input, string output)>(targetCount); // Pre-allocate capacity
        var random = Random.Shared;
        int safetyBreak = 0;
        const int maxSafetyBreak = 100000; // Increase safety break for larger target

        Console.WriteLine($"Generating training pairs (Target: {targetCount})...");

        // 1. Add Specific Crucial Pairs First
        AddSpecificPairs(pairs, targetCount);
        Console.WriteLine($"Added {pairs.Count} specific pairs initially.");

        // 2. Weighted Generation from Categories
        // Define weights (adjust these based on desired proportions)
        // Higher weight = higher chance of being selected
        var categoryWeights = new Dictionary<int, int> {
            {0, 15}, // Greetings (higher weight)
            {1, 5},  // Farewells
            {2, 10}, // Thanks
            {3, 20}, // Help/Meta (important for bot identity)
            {4, 15}, // Facts
            {5, 10}, // Jokes
            {6, 15}, // Capitals
            {7, 15}, // General Knowledge
            {8, 10}, // Short Interactions
            {9, 5}   // User Uncertainty
        };
        int totalWeight = categoryWeights.Sum(kv => kv.Value);

        while (pairs.Count < targetCount && safetyBreak < maxSafetyBreak)
        {
            // Weighted random selection
            int randomWeight = random.Next(totalWeight);
            int currentWeightSum = 0;
            int selectedCategory = -1;
            foreach (var kvp in categoryWeights)
            {
                currentWeightSum += kvp.Value;
                if (randomWeight < currentWeightSum)
                {
                    selectedCategory = kvp.Key;
                    break;
                }
            }

            if (selectedCategory == -1) { selectedCategory = 0; } // Fallback

            try
            {
                string inputBase = "";
                string outputBase = "";
                bool pairGenerated = false;

                switch (selectedCategory)
                {
                    case 0: // Greetings
                        if (greetings_in.Any() && greetings_out.Any()) {
                            inputBase = greetings_in[random.Next(greetings_in.Count)];
                            outputBase = greetings_out[random.Next(greetings_out.Count)];
                            pairGenerated = true;
                        }
                        break;
                    case 1: // Farewells
                         if (farewells_in.Any() && farewells_out.Any()) {
                            inputBase = farewells_in[random.Next(farewells_in.Count)];
                            outputBase = farewells_out[random.Next(farewells_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                    case 2: // Thanks
                         if (thanks_in.Any() && thanks_out.Any()) {
                            inputBase = thanks_in[random.Next(thanks_in.Count)];
                            outputBase = thanks_out[random.Next(thanks_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                    case 3: // Help/Meta
                         if (help_in.Any() && help_out.Any()) {
                            inputBase = help_in[random.Next(help_in.Count)];
                            outputBase = help_out[random.Next(help_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                    case 4: // Facts
                         if (facts_in.Any() && facts_out.Any()) {
                            inputBase = facts_in[random.Next(facts_in.Count)];
                            outputBase = facts_out[random.Next(facts_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                    case 5: // Jokes
                        if (jokes_in.Any() && jokes_out.Any()) {
                            inputBase = jokes_in[random.Next(jokes_in.Count)];
                            outputBase = jokes_out[random.Next(jokes_out.Count)];
                            pairGenerated = true;
                        }
                        break;
                    case 6: // Capitals
                        if (capitals.Any()) {
                            var randomCapitalPair = capitals.ElementAt(random.Next(capitals.Count));
                            inputBase = GenerateCapitalQuestion(randomCapitalPair.Key, random);
                            outputBase = randomCapitalPair.Value;
                            pairGenerated = true;
                        }
                        break;
                    case 7: // General Knowledge
                         if (general_knowledge_in_out.Any()) {
                            var randomGKPair = general_knowledge_in_out[random.Next(general_knowledge_in_out.Count)];
                            inputBase = randomGKPair.q;
                            outputBase = randomGKPair.a;
                            pairGenerated = true;
                         }
                        break;
                    case 8: // Short Interactions
                         if (short_interactions_in.Any() && short_interactions_out.Any()) {
                            inputBase = short_interactions_in[random.Next(short_interactions_in.Count)];
                            outputBase = short_interactions_out[random.Next(short_interactions_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                    case 9: // User Uncertainty
                         if (user_uncertainty_in.Any() && user_uncertainty_out.Any()) {
                            inputBase = user_uncertainty_in[random.Next(user_uncertainty_in.Count)];
                            outputBase = user_uncertainty_out[random.Next(user_uncertainty_out.Count)];
                            pairGenerated = true;
                         }
                        break;
                }

                // ***** AJUSTE: Adicionar EOS e adicionar à lista *****
                if (pairGenerated && !string.IsNullOrEmpty(inputBase) && !string.IsNullOrEmpty(outputBase))
                {
                    string outputWithEos = outputBase.Trim();
                    if (!outputWithEos.EndsWith(EosToken))
                    {
                        outputWithEos += EosToken;
                    }
                    pairs.Add((inputBase.Trim(), outputWithEos));
                }
                // ***** FIM AJUSTE *****

            }
            catch (ArgumentOutOfRangeException) { /* Ignore if lists are empty */ }
            safetyBreak++;
        }

        if (safetyBreak >= maxSafetyBreak)
        {
            Console.WriteLine($"Warning: Training data generation hit safety break at {pairs.Count} pairs.");
        }

        Console.WriteLine($"Generated {pairs.Count} total pairs before deduplication/shuffling.");

        // 3. Deduplicate and Shuffle
        var distinctPairs = pairs
            .GroupBy(p => new { p.input, p.output }) // Group by exact input/output pair
            .Select(g => g.First())                  // Take only the first occurrence
            .OrderBy(x => random.Next())             // Shuffle the distinct pairs
            .ToList();

        Console.WriteLine($"Returning {Math.Min(distinctPairs.Count, targetCount)} distinct and shuffled pairs.");

        // 4. Return exactly targetCount or less
        return distinctPairs.Take(targetCount).ToList();
    }

    // Helper to add specific, important pairs first
    private static void AddSpecificPairs(List<(string input, string output)> pairs, int targetCount)
    {
        // Helper function to add pair with EOS safely
        Action<string, string> AddPair = (input, output) => {
            if (pairs.Count < targetCount && !string.IsNullOrEmpty(input) && !string.IsNullOrEmpty(output)) {
                 string outputWithEos = output.Trim();
                 if (!outputWithEos.EndsWith(EosToken)) {
                     outputWithEos += EosToken;
                 }
                pairs.Add((input.Trim(), outputWithEos));
            }
        };

        AddPair("Who are you?", "I am Ninfa, a helpful chatbot created to assist you.");
        AddPair("What is your name?", "My name is Ninfa!");
        AddPair("Are you real?", "I am a computer program, existing as code.");
        AddPair("The capital of Netherlands", "Amsterdam is the constitutional capital, but The Hague is the seat of government.");
        AddPair("The capital of Bolivia", "Sucre is the constitutional capital, while La Paz is the seat of government.");
        AddPair("What's the weather like?", "I can't check the current weather for you, but I can share a fun fact!"); // Example negative
        AddPair("Thank you", "You're welcome!"); // Basic thanks
        AddPair("Bye", "Goodbye! It was nice chatting."); // Basic farewell
        // Add other crucial specific pairs here if needed
    }

     // Helper to generate varied capital questions
    private static string GenerateCapitalQuestion(string country, Random random)
    {
        int format = random.Next(5); // Increased formats
        switch (format)
        {
            case 0: return $"What is the capital of {country}?";
            case 1: return $"Capital of {country}";
            case 2: return $"{country} capital?";
            case 3: return $"Tell me the capital of {country}";
            case 4: return $"Which city is the capital of {country}?";
            default: return $"What is the capital of {country}?";
        }
    }
}