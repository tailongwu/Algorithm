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
    // 214
    // 最短回文串
    /*
        给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。
    */
    // 暴力求解：O(N^2)
    // KMP:O(N)
    string shortestPalindrome(string s)
    {
    }

    // 188
    // 买卖股票的最佳时机IV
    /*
        给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
        注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    */
    // 提示：dp[i][k][0/1]表示第i天最多交易k次没有/拥有股票的最大收益
    // dp[i][k][0]=max(dp[i-1][k][0], dp[i-1][k][1]+prices[i]);
    // dp[i][k][1]=max(dp[i-1][k][1], dp[i-1][k-1][0]-prices[i]);
    // 注意：未测试
    int maxProfit(int k, vector<int> &prices)
    {
        int len = prices.size();
        vector<vector<vector<int>>> dp(len + 1, vector<int>(k + 1, vector<int>(2, 0)));
        for (int i = 1; i <= len; i++)
        {
            for (int j = 0; j <= k; j++)
            {
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return max(dp[len][k][0], dp[len][k][1]);
    }

    // 174
    // 地下城游戏
    /*
        一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。
        骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。
        有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。
        为了尽快到达公主，骑士决定每次只向右或向下移动一步。
    */
    // 方法：如果往右走，只要在当前位置加了血或者扣了血后，等于dp[i][j+1]即可。那么dp[i][j+1]-[i,j]>=1
    int calculateMinimumHP(vector<vector<int>> &dungeon)
    {
        int row = dungeon.size();
        if (row == 0)
        {
            return 0;
        }
        int col = dungeon[0].size();
        if (col == 0)
        {
            return 0;
        }
        vector<vector<int>> dp(row + 1, vector<int>(col + 1, 0));
        dp[row - 1][col - 1] = dungeon[row - 1][col - 1] < 0 ? 1 - dungeon[row - 1][col - 1] : 1;
        for (int i = col - 2; i >= 0; i--)
        {
            dp[row - 1][i] = max(1, dp[row - 1][i + 1] - dungeon[row - 1][i]);
        }
        int ans = 0;
        for (int i = row - 2; i >= 0; i--)
        {
            dp[i][col - 1] = max(1, dp[i + 1][col - 1] - dungeon[i][col - 1]);
            for (int j = col - 2; j >= 0; j--)
            {
                int right = max(1, dp[i][j + 1] - dungeon[i][j]);
                int down = max(1, dp[i + 1][j] - dungeon[i][j]);
                dp[i][j] = min(right, down);
            }
        }
        return dp[0][0];
    }

    // 164
    // 最大间距
    /*
        给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。
        如果数组元素个数小于 2，则返回 0。
    */
    // 提示：桶排序思想，分在不同的桶里，每个桶有最大值和最小值，各个桶之间的间隔
    int maximumGap(vector<int> &nums)
    {
        sort(nums.begin(), nums.end());
        int len = nums.size();
        int ans = 0;
        for (int i = 1; i < len; i++)
        {
            ans = max(ans, nums[i] - nums[i - 1]);
        }
        return ans;
    }

    // 154
    // 寻找旋转排序数组中的最小值 II
    /*
        假设按照升序排序的数组在预先未知的某个点上进行了旋转。
        ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
        请找出其中最小的元素。
        注意数组中可能存在重复的元素。
        说明：
            这道题是 寻找旋转排序数组中的最小值 的延伸题目。
            允许重复会影响算法的时间复杂度吗？会如何影响，为什么？
    */
    // 提示：和最右边比较，如果大于最右边L=mid+1，如果小于最右边R=mid，否则R--
    int findMin(vector<int> &nums)
    {
        int L = 0, R = nums.size() - 1;
        while (L < R)
        {
            int mid = (L + R) >> 1;
            if (nums[mid] > nums[R])
            {
                L = mid + 1;
            }
            else if (nums[mid] < nums[R])
            {
                R = mid;
            }
            else
            {
                R--;
            }
        }
        return nums[L];
    }

    // 149
    // 直线上最多的点数
    /*
        给定一个二维平面，平面上有 n 个点，求最多有多少个点在同一条直线上。
    */
    // 提示：算每个点和其他点的斜率，hash存储斜率->个数，个数=n*(n-1)/2；注意小于3的点个数；注意重复点
    int maxPoints(vector<vector<int>> &points)
    {
        int len = points.size();
        if (len < 3)
        {
            return len;
        }
        map<pair<int, int>, int> m;
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            m.clear();
            int repeat = 0, result = 0;
            for (int j = i + 1; j < len; j++)
            {
                int x = points[i][0] - points[j][0];
                int y = points[i][1] - points[j][1];
                if (x == 0 && y == 0)
                {
                    repeat++;
                    continue;
                }
                int d = maxPoints_Gcd(x, y);
                if (d == 0)
                {
                    m[{x, y}]++;
                    result = max(result, m[{x, y}]);
                }
                else
                {
                    x /= d;
                    y /= d;
                    if (x < 0)
                    {
                        x = -x;
                        y = -y;
                    }
                    m[{x, y}]++;
                    result = max(result, m[{x, y}]);
                }
            }
            ans = max(ans, result + 1 + repeat);
        }
        return ans;
    }
    int maxPoints_Gcd(int a, int b)
    {
        return b == 0 ? a : maxPoints_Gcd(b, a % b);
    }

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

    // 140
    // 单词拆分II
    /*
        给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
        说明：
            分隔时可以重复使用字典中的单词。
            你可以假设字典中没有重复的单词。
    */
    // 提示：先判断是否有答案，再计算。
    vector<string> wordBreak(string s, vector<string> &wordDict)
    {
        int len = s.size();
        int dicLen = wordDict.size();
        vector<vector<string>> temp(len + 1);
        if (!wordBreak2(s, wordDict))
        {
            return temp[0];
        }
        vector<bool> dp(len + 1, false);
        dp[0] = true;
        for (int i = 0; i < len; i++)
        {
            if (dp[i])
            {
                for (int j = 0; j < dicLen; j++)
                {
                    int wordLen = wordDict[j].size();
                    if (i + wordLen > len)
                    {
                        continue;
                    }
                    string sub = s.substr(i, wordLen);
                    if (sub == wordDict[j])
                    {
                        dp[i + wordLen] = true;
                        for (int k = 0; k < temp[i].size(); k++)
                        {
                            string t = temp[i][k];
                            if (t.size() != 0)
                            {
                                t += ' ';
                            }
                            temp[i + wordLen].push_back(t + wordDict[j]);
                        }
                        if (temp[i].size() == 0)
                        {
                            temp[i + wordLen].push_back(wordDict[j]);
                        }
                    }
                }
            }
        }
        return temp[len];
    }
    bool wordBreak2(string s, vector<string> &wordDict)
    {
        int len = s.size();
        int wordSize = wordDict.size();
        vector<bool> dp(len + 5, false);
        dp[0] = true;
        for (int i = 0; i < len; i++)
        {
            if (dp[i])
            {
                for (int j = 0; j < wordSize; j++)
                {
                    if (i + wordDict[j].size() > len)
                    {
                        continue;
                    }
                    int index = i;
                    for (int k = 0; k < wordDict[j].size(); k++)
                    {
                        if (wordDict[j][k] != s[index])
                        {
                            break;
                        }
                        index++;
                    }
                    if (index - i == wordDict[j].size())
                    {
                        dp[index] = true;
                    }
                }
            }
        }
        return dp[len];
    }

    // 135
    // 分发糖果
    /*
        老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
        你需要按照以下要求，帮助老师给这些孩子分发糖果：
        每个孩子至少分配到 1 个糖果。
        相邻的孩子中，评分高的孩子必须获得更多的糖果。
        那么这样下来，老师至少需要准备多少颗糖果呢？
    */
    // 提示：从左往右一个数组left2right，从右往左一个数组right2left。最开始都是1，然后各自遍历，如果比前一个数大，那么就是left2right[i]=left2right[i-1]+1
    int candy(vector<int> &ratings)
    {
        int len = ratings.size();
        vector<int> left2right(len, 1);
        vector<int> right2left(len, 1);
        for (int i = 1; i < len; i++)
        {
            if (ratings[i] > ratings[i - 1])
            {
                left2right[i] = left2right[i - 1] + 1;
            }
        }
        for (int i = len - 2; i >= 0; i--)
        {
            if (ratings[i] > ratings[i + 1])
            {
                right2left[i] = right2left[i + 1] + 1;
            }
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            ans = ans + max(left2right[i], right2left[i]);
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
