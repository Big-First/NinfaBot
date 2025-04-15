namespace ChatBotAPI.Core;

public class TreeNode
{
    public List<int> InputSequence { get; set; } = new List<int>();
    public List<(List<int> Sequence, float Weight)> TargetSequences { get; set; } = new List<(List<int>, float)>();
    public List<Dictionary<int, float>> NextTokenProbs { get; set; } = new List<Dictionary<int, float>>();
    public bool IsDeterministic { get; set; } = false;
    public TreeNode? Left { get; set; }
    public TreeNode? Right { get; set; }
}