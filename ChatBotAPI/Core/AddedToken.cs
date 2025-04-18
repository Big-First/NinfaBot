namespace ChatBotAPI.Core;

public class AddedToken
{
    public AddedToken(){}
    public string content { get; set; }
    public bool single_word { get; set; }
    public bool lstrip { get; set; }
    public bool rstrip { get; set; }
    public bool normalized { get; set; }
    public int? special { get; set; }

    public AddedToken(string content, bool singleWord, bool lstrip, bool rstrip, bool normalized, int? special)
    {
        this.content = content;
        single_word = singleWord;
        this.lstrip = lstrip;
        this.rstrip = rstrip;
        this.normalized = normalized;
        this.special = special;
    }
}