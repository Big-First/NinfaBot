using System;
using System.Collections.Generic;
using System.Linq;

public static class TrainingDataGenerator // Changed class name for clarity
{
    // Method to generate the training data list
    public static List<(string input, string output)> GetTrainingData(int targetCount = 1500) // Added targetCount parameter
    {
        var pairs = new List<(string input, string output)>();
        var random = Random.Shared; // Use shared Random instance

        // --- Base Lists (Expanded) ---

        #region Greetings and Initial Interactions
        var greetings_in = new List<string> {
            "Hello", "Hi", "Hey", "Hi there", "Yo", "What's up?", "Howdy",
            "Good morning", "Good afternoon", "Good evening", "Greetings",
            "How are you?", "How's it going?", "Nice to see you", "Sup", "Heya"
        };
        var greetings_out = new List<string> {
            "Hi! How can I help you today?", "Hello! What is on your mind?", "Hey there! Nice to see you!",
            "Hello! How's everything?", "Hi! Ready to chat.", "Hey! What shall we talk about?",
            "Greetings! I'm at your service.", "Hi! Hope you're having a good day.",
            "Hello! Ask me anything.", "Good day! How can I assist?", // More generic
            "Hey! Nice to chat with you.", "Hi there! What can I do for you?"
        };
        #endregion

        #region Farewells
        var farewells_in = new List<string> {
            "Bye", "Goodbye", "See you later", "See ya", "Farewell", "Good night",
            "Talk to you later", "Catch you later", "I'm off", "Gotta go", "Take care", "Bye bye"
        };
        var farewells_out = new List<string> {
            "Goodbye! Have a great day!", "See you later! Come back soon.", "See ya!", "Farewell! It was nice chatting.",
            "Good night! Sleep well.", "Sure! Talk to you soon.", "Until next time!", "Catch you later!",
            "Bye bye! Stay safe!", "Take care too!"
        };
        #endregion

        #region Thanks
        var thanks_in = new List<string> {
            "Thank you", "Thanks", "Thx", "Cheers", "Grateful", "Appreciated",
            "Thanks a lot", "Many thanks", "Awesome, thanks", "Thank you so much", "Much obliged", "Thanks!"
        };
        var thanks_out = new List<string> {
            "You're welcome!", "No problem!", "Anytime!", "My pleasure!", "Don't mention it!",
            "Glad I could help!", "Happy to assist!", "You got it!", "It was nothing!", "Sure thing!"
        };
        #endregion

        #region Help Requests and Bot Meta Questions
        var help_in = new List<string> {
            "Help", "I need help", "Help me", "Can you help?", "Assistance please", "Need assistance",
            "What can you do?", "What are your capabilities?", "Who are you?", "What's your name?",
            "Tell me about yourself", "What is your purpose?", "How do you work?", "What can I ask you?",
            "Are you a robot?", "Are you real?", "Are you human?", "How old are you?", "Where do you live?",
            "Do you have feelings?", "Do you dream?", "Are you AI?", "Are you sentient?"
        };
        var help_out = new List<string> {
            "Sure, how can I help? Ask about facts, capitals, or tell me a joke!",
            "I'm here to help! What would you like to know?",
            "Absolutely! What's your question?",
            "I can answer general knowledge questions, tell facts, jokes, and chat about various topics.",
            "I am Ninfa, a helpful chatbot created to interact with you.", // Combined identity
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
        };
        #endregion

        #region Interesting Facts (Expanded Sample)
        var facts_in = new List<string> {
            "Tell me an interesting fact", "Give me a fun fact", "Random fact please",
            "Did you know?", "Something interesting", "General knowledge fact", "Teach me something new",
            "Fact of the day", "Any cool facts?"
        };
        var facts_out = new List<string> {
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
            // Add significantly more facts for better variety
        };
        #endregion

        #region Jokes (Expanded Sample)
        var jokes_in = new List<string> {
            "Tell me a joke", "Make me laugh", "Joke please", "Got any funny jokes?", "I want to laugh",
            "Tell me something funny", "Can you joke?", "Hit me with a joke"
        };
        var jokes_out = new List<string> {
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
            // Add significantly more jokes
        };
        #endregion

        #region Capitals (Expanded Sample - Use English country names)
        var capitals = new Dictionary<string, string> {
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
            // Add MANY more for comprehensive coverage
        };
        #endregion

        #region General Knowledge Questions (Expanded Sample)
        var general_knowledge_in_out = new List<(string q, string a)> {
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
            // Add MANY more
        };
        #endregion

        #region Short Interactions and Feedback (Expanded)
        var short_interactions_in = new List<string> {
            "Cool", "Nice", "Interesting", "Awesome", "Great", "Wow", "Amazing",
            "Ok", "Okay", "Alright", "Got it", "Understood", "Sure", "Fine",
            "Really?", "Seriously?", "Are you sure?", "For real?", "No way!",
            "Haha", "LOL", "LMAO", "Funny", "Good one!", "Hilarious", "That's funny",
            "You're smart", "Clever bot", "You know a lot", "Impressive",
            "You're funny", "I like you", "You're helpful", "Thanks for the info"
        };
        var short_interactions_out = new List<string> {
            "Glad you think so!", "Awesome!", "It is interesting, isn't it?", "Great!", "Indeed!",
            "Okay!", "Got it!", "Alright!", "Understood!", "Sure thing!", "Perfect!",
            "Yes, absolutely!", "Quite sure!", "For real!", "Believe it!",
            "Hehe!", "Glad I could make you chuckle!", "I try my best!", "Laughter is the best code!", "Glad you found it funny!",
            "Thank you! I process a lot of data.", "Thanks for the compliment!", "I'm always learning more!",
            "Thanks! Happy to be of service.", "How kind of you to say!", "Happy to be helpful!", "You're welcome!"
        };
        #endregion

        #region User Uncertainty and Boredom (Expanded)
        var user_uncertainty_in = new List<string> {
            "I don't know what to ask", "I'm out of ideas", "Suggest something", "Any suggestions?", "What else?",
            "I'm bored", "Anything fun?", "What else can we talk about?", "Give me a topic"
        };
        var user_uncertainty_out = new List<string> {
            "How about asking for the capital of a country?", "I can tell you an interesting fact, how about that?",
            "Want to hear a joke?", "Ask me about a planet or an animal!", "Why not ask about a famous scientist or artist?",
            "Let's learn something new! Ask about history or geography.",
            "How about a fun fact or a joke to liven things up?",
            "I can tell you about a country or a historical event.",
            "We can talk about technology, science, or just chat casually. What interests you?"
        };
        #endregion

        // --- Generation Logic ---

        Console.WriteLine($"Generating training pairs (Target: {targetCount})...");
        int safetyBreak = 0; // Prevent infinite loop if target is too high / lists too small

        // Prioritize specific examples first
        AddSpecificPairs(pairs, targetCount); // Add specific pairs like identity, complex capitals, etc.

        // Then generate from categories until target is reached
        while (pairs.Count < targetCount && safetyBreak < targetCount * 5) // Add safety break
        {
            int category = random.Next(10); // Random category selection
            try
            {
                switch (category)
                {
                    case 0: // Greetings (Generate multiple variants)
                        string greetIn = greetings_in[random.Next(greetings_in.Count)];
                        string greetOut = greetings_out[random.Next(greetings_out.Count)];
                        pairs.Add((greetIn, greetOut));
                        if (random.NextDouble() < 0.5 && pairs.Count < targetCount) // Occasionally add another variant
                             pairs.Add((greetIn, greetings_out[random.Next(greetings_out.Count)]));
                        break;

                    case 1: // Farewells
                        pairs.Add((farewells_in[random.Next(farewells_in.Count)], farewells_out[random.Next(farewells_out.Count)]));
                        break;

                    case 2: // Thanks
                        pairs.Add((thanks_in[random.Next(thanks_in.Count)], thanks_out[random.Next(thanks_out.Count)]));
                        break;

                    case 3: // Help/Meta
                        pairs.Add((help_in[random.Next(help_in.Count)], help_out[random.Next(help_out.Count)]));
                        break;

                    case 4: // Facts
                        pairs.Add((facts_in[random.Next(facts_in.Count)], facts_out[random.Next(facts_out.Count)]));
                        break;

                    case 5: // Jokes
                        pairs.Add((jokes_in[random.Next(jokes_in.Count)], jokes_out[random.Next(jokes_out.Count)]));
                        break;

                    case 6: // Capitals (Generate random question format)
                        var randomCapitalPair = capitals.ElementAt(random.Next(capitals.Count));
                        string capQuestion = GenerateCapitalQuestion(randomCapitalPair.Key, random);
                        pairs.Add((capQuestion, randomCapitalPair.Value));
                        break;

                    case 7: // General Knowledge
                        var randomGKPair = general_knowledge_in_out[random.Next(general_knowledge_in_out.Count)];
                        pairs.Add((randomGKPair.q, randomGKPair.a));
                        break;

                    case 8: // Short Interactions
                        pairs.Add((short_interactions_in[random.Next(short_interactions_in.Count)], short_interactions_out[random.Next(short_interactions_out.Count)]));
                        break;

                     case 9: // User Uncertainty
                        pairs.Add((user_uncertainty_in[random.Next(user_uncertainty_in.Count)], user_uncertainty_out[random.Next(user_uncertainty_out.Count)]));
                        break;
                }
            }
            catch (ArgumentOutOfRangeException) { /* Ignore if lists are empty, though they shouldn't be */ }
            safetyBreak++;
        }

        Console.WriteLine($"Generated {pairs.Count} unique pairs before potential trimming/shuffling.");

        // Optional: Shuffle for potentially better training dynamics if needed
        var shuffledPairs = pairs.OrderBy(x => random.Next()).ToList();

        // Return exactly targetCount or less
        return shuffledPairs.Take(targetCount).ToList();
    }

    // Helper to add specific, important pairs first
    private static void AddSpecificPairs(List<(string input, string output)> pairs, int targetCount)
    {
         if (pairs.Count >= targetCount) return;
         pairs.Add(("Who are you?", "I am Ninfa, a helpful chatbot created to assist you."));
         if (pairs.Count >= targetCount) return;
         pairs.Add(("What is your name?", "My name is Ninfa!"));
         if (pairs.Count >= targetCount) return;
         pairs.Add(("Are you real?", "I am a computer program, existing as code."));
         if (pairs.Count >= targetCount) return;
         pairs.Add(("The capital of Netherlands", "Amsterdam is the constitutional capital, but The Hague is the seat of government."));
         if (pairs.Count >= targetCount) return;
         pairs.Add(("The capital of Bolivia", "Sucre is the constitutional capital, while La Paz is the seat of government."));
         // Add other crucial specific pairs here
    }

     // Helper to generate varied capital questions
    private static string GenerateCapitalQuestion(string country, Random random)
    {
        int format = random.Next(4);
        switch (format)
        {
            case 0: return $"What is the capital of {country}?";
            case 1: return $"Capital of {country}";
            case 2: return $"{country} capital?";
            case 3: return $"Tell me the capital of {country}";
            default: return $"What is the capital of {country}?";
        }
    }
}