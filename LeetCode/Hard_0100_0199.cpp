#include <map>
#include <cmath>
#include <queue>
#include <stack>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
struct Node3
{
    int val;
    Node3 *next;
    Node3 *random;
    Node3() {}
    Node3(int _val, Node3 *_next, Node3 *_random)
    {
        val = _val;
        next = _next;
        random = _random;
    }
};
struct Node2
{
    int val;
    vector<Node2 *> neighbors;
    Node2() {}
    Node2(int _val, vector<Node2 *> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }
};
struct Node
{
    int val;
    Node *left;
    Node *right;
    Node *next;
    Node(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution
{
public:
    // 145
    // 二叉树的后序遍历
    /*
        给定一个二叉树，返回它的 后序 遍历。
        进阶: 递归算法很简单，你可以通过迭代算法完成吗？
    */
    vector<int> postorderTraversal(TreeNode *root)
    {
        // 方法1：
        /*
            vector<int> ans;
            if (root == 0)
            {
                return ans;
            }
            stack<TreeNode*> sta;
            TreeNode *node, *pre = 0;
            sta.push(root);
            while (!sta.empty())
            {
                node = sta.top();
                if (pre == 0 || pre->left == node || pre->right == node)
                {
                    // 上一个节点是父节点，先入左儿子后入右儿子
                    if (node->left != 0)
                    {
                        sta.push(node->left);
                    }
                    else if (node->right != 0)
                    {
                        sta.push(node->right);
                    }
                }
                else if (pre == node->left)
                {
                    // 上一个节点是左儿子，入右儿子
                    if (node->right != 0)
                    {
                        sta.push(node->right);
                    }
                }
                else
                {
                    ans.push_back(node->val);
                    sta.pop();
                }
                pre = node;
            }
            return ans;
        */

        // 方法2：先入所有右儿子，再弹出再入左儿子
        vector<int> ans;
        if (root == 0)
        {
            return ans;
        }
        stack<TreeNode *> sta1;
        stack<TreeNode *> sta2;
        TreeNode *node = root;
        while (true)
        {
            while (node)
            {
                sta1.push(node);
                sta2.push(node);
                node = node->right;
            }
            if (!sta1.empty())
            {
                node = sta1.top();
                sta1.pop();
                node = node->left;
            }
            else
            {
                break;
            }
        }
        while (!sta2.empty())
        {
            node = sta2.top();
            sta2.pop();
            ans.push_back(node->val);
        }
        return ans;
    }

    // 132
    // 分割回文串II
    /*
        给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
        返回符合要求的最少分割次数。
    */
    // 提示：用p[i][j]保存是否[i...j]是回文串。dp[i]代表[i...n-1]至少要分割多少次。
    // 注意：dp[len]=-1。
    int minCut(string s)
    {
        int len = s.size();
        vector<int> dp(len + 1);
        vector<vector<bool>> p(len, vector<bool>(len));
        for (int i = 0; i < len; i++)
        {
            p[i][i] = true;
        }
        dp[len] = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            dp[i] = INT_MAX;
            for (int j = i; j < len; j++)
            {
                if (s[i] == s[j] && (i + 1 > j - 1 || p[i + 1][j - 1]))
                {
                    p[i][j] = true;
                    dp[i] = min(dp[i], 1 + dp[j + 1]);
                }
            }
        }
        return dp[0];
    }

    // 128
    // 最长连续序列
    /*
        定一个未排序的整数数组，找出最长连续序列的长度。
        要求算法的时间复杂度为 O(n)。
    */
    // 提示：以每个数作为开始，看nums[i]+1,nums[i]+2...是否在数组。
    // 计算过的nums[i]的长度保存下来
    int longestConsecutive(vector<int> &nums)
    {
        int len = nums.size();
        map<int, int> m;
        vector<int> numLen(len, -1);
        for (int i = 0; i < len; i++)
        {
            m[nums[i]] = i + 1;
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            int cnt = 0, k = nums[i];
            while (m[k] != 0)
            {
                if (numLen[m[k] - 1] != -1)
                {
                    cnt += numLen[m[k] - 1];
                    break;
                }
                cnt++;
                k++;
            }
            numLen[i] = cnt;
            ans = max(ans, cnt);
        }
        return ans;
    }

    // 126
    // 单词接龙II
    /*
        给定两个单词（beginWord 和 endWord）和一个字典 wordList，找出所有从 beginWord 到 endWord 的最短转换序列。转换需遵循如下规则：

        每次转换只能改变一个字母。
        转换过程中的中间单词必须是字典中的单词。
        说明:
            如果不存在这样的转换序列，返回一个空列表。
            所有单词具有相同的长度。
            所有单词只由小写字母组成。
            字典中不存在重复的单词。
            你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
    */
    // 提示：双向BFS

    // 124
    // 二叉树中的最大路径和
    /*
        给定一个非空二叉树，返回其最大路径和。
        本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
    */
    // 提示：可以求出每个节点左子树的最长路径，右子树的最长路径，左右路径相加再加该节点，就是最大值。
    int maxPathSum(TreeNode *root)
    {
        if (root == 0)
        {
            return 0;
        }
        int ans = root->val;
        DFS_maxPathSum(root, ans);
        return ans;
    }
    int DFS_maxPathSum(TreeNode *root, int &ans)
    {
        if (root == 0)
        {
            return 0;
        }
        int left = max(0, DFS_maxPathSum(root->left, ans));
        int right = max(0, DFS_maxPathSum(root->right, ans));
        ans = max(ans, left + right + root->val);
        return max(left, right) + root->val;
    }

    // 123
    // 买卖股票的最佳时机III
    /*
        给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
        注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    */
    // 方法1：O(N^2)，以i为分界点，左边最多交易一次的最大获利，右边最多交易一次最大获利
    // 方法2：O(N)，left[i]保存从0-i的最多一次交易的最大值。right[i]保存从i到n-1的最多一次交易的最大值。
    int maxProfit(vector<int> &prices)
    {
        int len = prices.size();
        vector<int> left(len);
        vector<int> right(len);
        int mi = INT_MAX;
        for (int i = 0; i < len; i++)
        {
            mi = min(mi, prices[i]);
            left[i] = prices[i] - mi;
            if (i > 0 && left[i] < left[i - 1])
            {
                left[i] = left[i - 1];
            }
        }
        int ma = INT_MIN;
        for (int i = len - 1; i >= 0; i--)
        {
            ma = max(ma, prices[i]);
            right[i] = ma - prices[i];
            if (i < len - 1 && right[i] < right[i + 1])
            {
                right[i] = right[i + 1];
            }
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            ans = max(ans, (i == 0 ? 0 : left[i - 1]) + right[i]);
        }
        return ans;
    }

    // 115
    // 不同的子序列
    /*
        给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。
        一个字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
    */
    // 提示：dp[i][j]，表示由s的前i个有多少个t的前j个。如果s[i-1]==t[j-1]，那么dp[i][j]=dp[i-1][j-1]+dp[i-1][j];否则dp[i][j]=d[i-1][j]；
    // 所有dp[i][0]=1
    int numDistinct(string s, string t)
    {
        int len1 = s.size(), len2 = t.size();
        vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1, 0));
        for (int i = 0; i <= len1; i++)
        {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= len1; i++)
        {
            for (int j = 1; j <= len2; j++)
            {
                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                }
                else
                {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[len1][len2];
    }
};
int main()
{
    Solution *solution = new Solution();
    return 0;
}
