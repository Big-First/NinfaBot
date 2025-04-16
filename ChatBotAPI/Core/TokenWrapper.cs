namespace ChatBotAPI.Core
{
    public class TokenWrapper
    {
        public int[] Token { get; set; }

        public TokenWrapper(int[] token)
        {
            Token = token;
        }
    }
}