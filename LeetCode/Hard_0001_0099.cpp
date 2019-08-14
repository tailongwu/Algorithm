#include<map>
#include<cmath>
#include<queue>
#include<stack>
#include<cstdio>
#include<cstring>
#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;
struct Node3
{
    int val;
    Node3* next;
    Node3* random;
    Node3() {}
    Node3(int _val, Node3* _next, Node3* _random)
    {
        val = _val;
        next = _next;
        random = _random;
    }
};
struct Node2
{
    int val;
    vector<Node2*> neighbors;
    Node2() {}
    Node2(int _val, vector<Node2*> _neighbors)
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
    // 85
    // 最大矩形
    /*
        给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
    */
    int maximalRectangle(vector<vector<char> >& matrix)
    {
        int ans = 0;
        int row = matrix.size();
        if (row == 0)
        {
            return 0;
        }
        int col = matrix[0].size();
        if (col == 0)
        {
            return 0;
        }
        vector<vector<int> > width(row + 1, vector<int> (col + 1, 0));
        vector<vector<int> > height(row + 1, vector<int> (col + 1, 0));
        for (int i = 1; i <= row; i++)
        {
            for (int j = 1; j <= col; j++)
            {
                if (matrix[i - 1][j - 1] == '1')
                {
                    int w, h;
                    if (width[i][j - 1] - 1 >= width[i - 1][j])
                    {
                        if (height[i][j - 1] - 1 >= height[i - 1][j])
                        {
                            w = width[i][j - 1] + 1;
                            h = height[i - 1][j] + 1;
                        }
                        else
                        {
                            int w1, w2, h1, h2;
                            w1 = width[i][j - 1] + 1;
                            h1 = height[i][j - 1];
                            w2 = width[i - 1][j];
                            h2 = htight[i - 1][j] + 1;
                            if (w1 * h1 > w2 * h2)
                            {
                                w = w1;
                                h = h1;
                            }
                            else
                            {
                                w = w2;
                                h = h2;
                            }
                        }
                    }
                    else
                    {
                        if (height[i][j - 1] - 1 >= height[i - 1][j])
                        {
                            w = width[i][j - 1] + 1;
                            h = height[i - 1][j] + 1;
                        }
                        else
                        {
                            w = width[i][j - 1] + 1;

                        }
                    }
                    ans = max(ans, width[i][j] * height[i][j]);
                }
                cout << i << " " << j << "  "<< width[i][j] << " " << height[i][j]<<endl;
            }
        }
        return ans;
    }


    // 84
    // 柱状图中最大的矩形
    /*
        给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
        求在该柱状图中，能够勾勒出来的矩形的最大面积。
    */
    // 提示：单调栈（栈内是1，3，5，如果4要入栈，那么先弹出5，再4入栈。）。如果当前高度大于栈顶，入栈。如果当前高度小于栈顶A，栈顶A出栈，并以A为高度，最左边为当前栈顶元素，最右边为当前高度的左边。
    // 比如：2,1,5,6,2,3. 2入栈；1会让2出栈同时2为高度的最左边可扩展到当前栈顶右边可扩展到当前1的前一个；5入栈；6入栈；2会让6出栈同时6的左边为5的下一个右边为2的上一个，2会让5出栈同时5的左边为5的前一个右边为2的上一个。
    /*
        单调队列题目
        输入一个长度为n的整数序列，从中找出一段不超过M的连续子序列，使得整个序列的和最大。
        设sum[i]为前i个数的和，要求j>i&&j-i<m&&sum[j]-sum[i]最大值。
        假设当前遍历到第i个，之前有sum[j]和sum[k]并且j<k并且sum[j]>sum[k]，肯定选择sum[k]更优。
        while(!Q.empty()&&Q.front()<i-m)Q.pop_front();
		ans=max(ans,s[i]-s[Q.front()]);
		while(!Q.empty()&&s[Q.back()]>=s[i])Q.pop_back();
		Q.push_back(i)
     */
    int largestRectangleArea(vector<int>& heights)
    {
        heights.push_back(0);
        int len = heights.size(), ans = 0, top = 0;
        stack<int> sta;
        for (int i = 0; i < len; i++)
        {
            while (!sta.empty() && heights[i] < heights[sta.top()])
            {
                top = sta.top();
                sta.pop();
                ans = max(ans, heights[top] * (i - (sta.empty() ? 0 : sta.top() + 1)));
            }
            sta.push(i);
        }
        return ans;
    }


    // 76
    // 最小覆盖子串
    /*
        给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。
        示例：
            输入: S = "ADOBECODEBANC", T = "ABC"
            输出: "BANC"
        说明：
            如果 S 中不存这样的子串，则返回空字符串 ""。
            如果 S 中存在这样的子串，我们保证它是唯一的答案。
    */
    // 提示：滑动窗口，"a","aa"为false
    string minWindow(string s, string t)
    {
        int sLen = s.size(), tLen = t.size(), posLen;
        int left, ansL = -1, ansR = sLen, tDiffCount = 0, sDiffCount = 0;
        string ans = "";
        vector<int> tVis(256, 0);
        vector<int> sVis(256, 0);
        vector<int> pos;
        for (int i = 0; i < tLen; i++)
        {
            if (tVis[t[i]] == 0)
            {
                tDiffCount++;
            }
            tVis[t[i]]++;
        }
        for (int i = 0; i < sLen; i++)
        {
            pos.push_back(i);
        }
        posLen = pos.size();
        if (posLen == 0)
        {
            return ans;
        }
        left = 0;
        for (int i = 0; i < posLen; i++)
        {
            sVis[s[pos[i]]]++;
            if (sVis[s[pos[i]]] == tVis[s[pos[i]]])
            {
                sDiffCount++;
            }
            if (sDiffCount == tDiffCount)
            {
                if (pos[i] - pos[left] < ansR - ansL)
                {
                    ansR = pos[i];
                    ansL = pos[left];
                }
                while (left < i)
                {
                    sVis[s[pos[left]]]--;
                    if (sVis[s[pos[left]]] < tVis[s[pos[left]]])
                    {
                        sDiffCount--;
                    }
                    left++;
                    if (sDiffCount != tDiffCount)
                    {
                        break;
                    }
                    if (pos[i] - pos[left] < ansR - ansL)
                    {
                        ansR = pos[i];
                        ansL = pos[left];
                    }
                }
            }
        }
        if (ansL != -1)
        {
            for (int i = ansL; i <= ansR; i++)
            {
                ans += s[i];
            }
        }
        return ans;
    }


    // 72
    // 编辑距离
    /*
        给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。
        你可以对一个单词进行如下三种操作：
            插入一个字符
            删除一个字符
            替换一个字符
    */
    // 提示：dp[i][j]表示从长度为i的字符变成长度为j需要的最小操作数。dp[i][j]=(words1[i] == words[j]) ? dp[i-1][j-1] : 1+min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1]))
    int minDistance(string word1, string word2)
    {
        int len1 = word1.size(), len2 = word2.size();
        vector<vector<int> > dp(len1 + 1, vector<int> (len2 + 1));
        for (int i = 0; i <= len2; i++)
        {
            dp[0][i] = i;
        }
        for (int i = 0; i <= len1; i++)
        {
            dp[i][0] = i;
        }
        for (int i = 1; i <= len1; i++)
        {
            for (int j = 1; j <= len2; j++)
            {
                if (word1[i - 1] == word2[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else
                {
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]));
                }
            }
        }
        return dp[len1][len2];
    }


    // 68
    // 文本左右对齐
    /*
        给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
        你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
        要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
        文本的最后一行应为左对齐，且单词之间不插入额外的空格。
        说明:
            单词是指由非空格字符组成的字符序列。
            每个单词的长度大于 0，小于等于 maxWidth。
            输入单词数组 words 至少包含一个单词。
    */
    vector<string> fullJustify(vector<string>& words, int maxWidth)
    {
        vector<string> ans;
        int len = words.size(), i = 0;
        while (i < len)
        {
            int sum = 0, j = i, wordsSum = 0;
            while (j < len)
            {
                sum += words[j].size();
                wordsSum += words[j].size();
                if (j != i)
                {
                    sum++;
                }
                if (sum > maxWidth)
                {
                    wordsSum -= words[j].size();
                    j--;
                    break;
                }
                j++;
            }
            if (j == len)
            {
                j--;
            }
            // [i,j]
            int spaceCount = maxWidth - wordsSum;
            if (i == j)
            {
                string result = words[i];
                for (int k = 0; k < spaceCount; k++)
                {
                    result += ' ';
                }
                ans.push_back(result);
                i = j + 1;
                continue;
            }
            int r = spaceCount % (j - i);
            int avg = spaceCount / (j - i);
            string result = "";
            for (int k = i; k < j; k++)
            {
                result += words[k];
                for (int o = 0; o < avg; o++)
                {
                    result += ' ';
                }
                if (r > 0)
                {
                    result += ' ';
                    r--;
                }
            }
            result += words[j];
            ans.push_back(result);
            i = j + 1;
        }
        // 处理最后一行
        len = ans.size();
        string last = "";
        int index = 0;
        for (int i = 0; i < maxWidth; i++)
        {
            last += ' ';
        }
        for (int i = 0; i < maxWidth; i++)
        {
            if (ans[len - 1][i] == ' ')
            {
                if (last[index - 1] == ' ')
                {
                    continue;
                }
                index++;
            }
            else
            {
                last[index++] = (char)ans[len - 1][i];
            }
        }
        ans[len - 1] = last;
        return ans;
    }


    // 65
    // 有效数字
    /*
        验证给定的字符串是否可以解释为十进制数字。

        例如:
        "0" => true
        " 0.1 " => true
        "abc" => false
        "1 a" => false
        "2e10" => true
        " -90e3   " => true
        " 1e" => false
        "e3" => false
        " 6e-1" => true
        " 99e2.5 " => false
        "53.5e93" => true
        " --6 " => false
        "-+3" => false
        "95a54e53" => false
        说明: 我们有意将问题陈述地比较模糊。在实现代码之前，你应当事先思考所有可能的情况。这里给出一份可能存在于有效十进制数字中的字符列表：
        数字 0-9
        指数 "e"
        正/负号 "+"/"-"
        小数点 "."
        当然，在输入中，这些字符的上下文也很重要。
    */
    bool isNumber(string s)
    {

    }


    // 57
    // 插入区间
    /*
        给出一个无重叠的 ，按照区间起始端点排序的区间列表。
        在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
    */
    // 注意：各自为空数组，没有交集(在前在后在中间)
    vector<vector<int> > insert(vector<vector<int> >& intervals, vector<int>& newInterval)
    {
        int len = intervals.size();
        vector<vector<int> > ans;
        if (len == 0)
        {
            ans.push_back(newInterval);
            return ans;
        }
        if (len == 0)
        {
            return intervals;
        }
        for (int i = 0; i <= len; i++)
        {
            if ((i == 0 || intervals[i - 1][1] < newInterval[0]) && (i == len || intervals[i][0] > newInterval[1]))
            {
                ans.push_back(newInterval);
            }
            if (i == len)
            {
                continue;
            }
            if (intervals[i][1] < newInterval[0] || intervals[i][0] > newInterval[1])
            {
                ans.push_back(intervals[i]);
            }
            else
            {
                int sta = min(intervals[i][0], newInterval[0]);
                while (i < len && intervals[i][0] <= newInterval[1])
                {
                    i++;
                }
                i--;
                int en = max(intervals[i][1], newInterval[1]);
                vector<int> p(2);
                p[0] = sta;
                p[1] = en;
                ans.push_back(p);
            }
        }
        return ans;
    }


    // 52
    // N皇后 II
    /*
        n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
        给定一个整数 n，返回 n 皇后不同的解决方案的数量。
    */
    // 提示：可换成二进制保存而不是字符串
    int totalNQueens(int n)
    {
        int ans = 0;
        string s = "";
        for (int i = 0; i < n; i++)
        {
            s += '.';
        }
        vector<string> result(n, s);
        DFS_solveNQueens(ans, result, n, 0);
        return ans;
    }
    void DFS_solveNQueens(int &ans, vector<string> &result, int n, int row)
    {
        if (row == n)
        {
            ans++;
            return;
        }
        for (int i = 0; i < n; i++)
        {
            bool exist = false;
            // 检查列
            for (int j = 0; j < row && !exist; j++)
            {
                if (result[j][i] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            // 检查左对角线
            for (int j = row - 1; j >= 0 && !exist; j--)
            {
                if (i - (row - i - j) - 1 < 0)
                {
                    break;
                }
                if (result[j][i - (row - 1 - j) - 1] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            // 检查右对角线
            for (int j = row - 1; j >= 0 && !exist; j--)
            {
                if (i + (row - 1 - j) + 1 >= n)
                {
                    break;
                }
                if (result[j][i + (row - 1 - j) + 1] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            if (!exist)
            {
                result[row][i] = 'Q';
                DFS_solveNQueens(ans, result, n, row + 1);
                result[row][i] = '.';
            }
        }
    }


    // 51
    // N皇后
    /*
        n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
        给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
        每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
    */
    vector<vector<string> > solveNQueens(int n)
    {
        vector<vector<string> > ans;
        string s = "";
        for (int i = 0; i < n; i++)
        {
            s += '.';
        }
        vector<string> result(n, s);
        DFS_solveNQueens(ans, result, n, 0);
        return ans;
    }
    void DFS_solveNQueens(vector<vector<string> > &ans, vector<string> &result, int n, int row)
    {
        if (row == n)
        {
            ans.push_back(result);
            return;
        }
        for (int i = 0; i < n; i++)
        {
            bool exist = false;
            // 检查列
            for (int j = 0; j < row && !exist; j++)
            {
                if (result[j][i] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            // 检查左对角线
            for (int j = row - 1; j >= 0 && !exist; j--)
            {
                if (i - (row - i - j) - 1 < 0)
                {
                    break;
                }
                if (result[j][i - (row - 1 - j) - 1] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            // 检查右对角线
            for (int j = row - 1; j >= 0 && !exist; j--)
            {
                if (i + (row - 1 - j) + 1 >= n)
                {
                    break;
                }
                if (result[j][i + (row - 1 - j) + 1] == 'Q')
                {
                    exist = true;
                    break;
                }
            }
            if (!exist)
            {
                result[row][i] = 'Q';
                DFS_solveNQueens(ans, result, n, row + 1);
                result[row][i] = '.';
            }
        }
    }


    // 45
    // 跳跃游戏II
    /*
        给定一个非负整数数组，你最初位于数组的第一个位置。
        数组中的每个元素代表你在该位置可以跳跃的最大长度。
        你的目标是使用最少的跳跃次数到达数组的最后一个位置。
        说明：
            假设你总是可以到达数组的最后一个位置。
    */
    // 提示：只有超过之前的最大值才step加一
    int jump(vector<int>& nums)
    {
        int len = nums.size();
        int ans = 0, maxpos = 0, curpos = 0;
        for (int i = 0; i < len; i++)
        {
            if (i > maxpos)
            {
                return -1;
            }
            if (i > curpos)
            {
                curpos = maxpos;
                ans++;
            }
            maxpos = max(maxpos, i + nums[i]);
        }
        return ans;
    }


    // 44
    // 通配符匹配
    /*
        给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
        '?' 可以匹配任何单个字符。
        '*' 可以匹配任意字符串（包括空字符串）。
        两个字符串完全匹配才算匹配成功。
        说明:
            s 可能为空，且只包含从 a-z 的小写字母。
            p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
    */
    bool isMatchII(string s, string p)
    {
        int len1 = s.size(), len2 = p.size();
        vector<vector<bool> > dp(len1 + 1, vector<bool> (len2, false));
        dp[0][0] = true;
        for (int i = 1; i <= len2; i++)
        {
            // 有其它字符就是false
            if (p[j] == '*')
            {
                dp[0][j] = dp[0][j - 1];
            }
        }
        for (int i = 1; i <= len1; i++)
        {
            for (int j = 1; j <= len2; j++)
            {
                if (s[i - 1] == p[j - 1] || p[j - 1] == '?')
                {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else if (p[j - 1] == '*')
                {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
    }


    // 42
    // 接雨水
    /*
        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
    */
    // 提示：找到最高点，然后最左向最高点遍历，最右向最高点遍历
    int trap(vector<int>& height)
    {
        int len = height.size();
        int ma = -1, index = -1;
        for (int i = 0; i < len; i++)
        {
            if (height[i] > ma)
            {
                ma = height[i];
                index = i;
            }
        }
        int ans = 0, tmp = -1;
        for (int i = 0; i < index; i++)
        {
            if (height[i] > tmp)
            {
                tmp = height[i];
            }
            else
            {
                ans = ans + tmp - height[i];
            }
        }
        tmp = -1;
        for (int i = len - 1; i > index; i--)
        {
            if (height[i] > tmp)
            {
                tmp = height[i];
            }
            else
            {
                ans = ans + tmp - height[i];
            }
        }
        return ans;
    }


    // 41
    // 缺失的第一个正数
    /*
        给定一个未排序的整数数组，找出其中没有出现的最小的正整数。
        说明:
            你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。
    */
    // 提示：桶排序思想，每个数应该放到对应的位置。比如3,4,-1,1->-1,4,3,1->-1,1,3,4。最后再扫描空缺部分。
    int firstMissingPositive(vector<int>& nums)
    {
        int len = nums.size();
        for (int i = 0; i < len; i++)
        {
            if (nums[i] >= 1 && nums[i] <= len && nums[i] != i + 1 && nums[nums[i] - 1] != nums[i])
            {
                swap(nums[i], nums[nums[i] - 1]);
                i--;
            }
        }
        for (int i = 0; i < len; i++)
        {
            if (nums[i] != i + 1)
            {
                return i + 1;
            }
        }
        return len + 1;
    }


    // 37
    // 解数独
    /*
        编写一个程序，通过已填充的空格来解决数独问题。
        一个数独的解法需遵循如下规则：
            数字 1-9 在每一行只能出现一次。
            数字 1-9 在每一列只能出现一次。
            数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
            空白格用 '.' 表示。
        Note:
            给定的数独序列只包含数字 1-9 和字符 '.' 。
            你可以假设给定的数独只有唯一解。
            给定数独永远是 9x9 形式的。
    */
    bool ans_solveSudoku = false;
    void solveSudoku(vector<vector<char> >& board)
    {
        if (ans_solveSudoku)
        {
            return;
        }
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] == '.')
                {
                    for (int k = 1; k < 10; k++)
                    {
                        bool exist = false;
                        int row = i / 3 * 3;
                        int col = j / 3 * 3;
                        for (int r = 0; r < 9; r++)
                        {
                            if (board[i][r] == k + '0')
                            {
                                exist = true;
                                break;
                            }
                            if (board[r][j] == k + '0')
                            {
                                exist = true;
                                break;
                            }
                            if (board[row + r / 3][col + r % 3] == k + '0')
                            {
                                exist = true;
                                break;
                            }
                        }
                        if (!exist)
                        {
                            board[i][j] = k + '0';
                            solveSudoku(board);
                            if (ans_solveSudoku)
                            {
                                return;
                            }
                            board[i][j] = '.';
                        }
                    }
                    return;
                }
            }
        }
        ans_solveSudoku = true;
        return;
    }


    // 32
    // 最长有效括号
    /*
        给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
    */
    // 提示：(()这种情况，所以需要倒着再计算一次
    int longestValidParentheses(string s)
    {
        int len = s.size();
        int cnt = 0, ans1 = 0, ans2 = 0, sta = 0;
        for (int i = 0; i < len; i++)
        {
            if (s[i] == '(')
            {
                cnt++;
            }
            else
            {
                cnt--;
            }
            if (cnt < 0)
            {
                sta = i + 1;
                cnt = 0;
            }
            if (cnt == 0)
            {
                ans1 = max(ans1, i - sta + 1);
            }
        }
        sta = len - 1;
        cnt = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            if (s[i] == ')')
            {
                cnt++;
            }
            else
            {
                cnt--;
            }
            if (cnt < 0)
            {
                sta = i - 1;
                cnt = 0;
            }
            if (cnt == 0)
            {
                ans2 = max(ans2, sta - i + 1);
            }
        }
        return max(ans1, ans2);
    }


    // 30
    // 串联所有单词的子串
    /*
        给定一个字符串 s 和一些长度相同的单词 words。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
        注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。
    */
    vector<int> findSubstring(string s, vector<string>& words)
    {
        vector<int> ans;
        int len1 = s.size(), len2 = words.size(), len3 = words[0].size();
        for (int i = 0; i + len2 * len3 < len1; i++)
        {
            map<int, int> m;
            for (int j = 0; j < len2; j++)
            {

            }
        }
        return ans;
    }


    // 25
    // K 个一组翻转链表
    /*
        给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
        k 是一个正整数，它的值小于或等于链表的长度。
        如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
        示例 :
        给定这个链表：1->2->3->4->5
        当 k = 2 时，应当返回: 2->1->4->3->5
        当 k = 3 时，应当返回: 3->2->1->4->5
    */
    ListNode* reverseKGroup(ListNode* head, int k)
    {
        if (head == 0)
        {
            return head;
        }
        int index = 0;
        ListNode *newHead = head, *node = head, *next, *first = 0;
        while (node != 0 && index < k)
        {
            node = node->next;
            index++;
        }
        if (index < k)
        {
            return head;
        }
        node = head;
        index = 0;
        while (index < k && node != 0)
        {
            if (first == 0)
            {
                first = node;
            }
            next = node->next;
            node->next = newHead;
            newHead = node;
            node = next;
            index++;
        }
        if (first != 0)
        {
            first->next = reverseKGroup(node, k);;
        }
        else
        {
            return 0;
        }
        return newHead;
    }


    // 23
    // 合并K个排序链表
    /*
        合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。
    */
    // 提示：方法1：两两合并，归并思想；方法2：优先队列维护topk，每次弹出表示哪个链表最小
    ListNode* mergeKLists(vector<ListNode*>& lists)
    {
        if (lists.size() == 0)
        {
            return 0;
        }
        Do_mergeKLists(lists, 0, lists.size() - 1);
        return lists[0];
    }
    void Do_mergeKLists(vector<ListNode*>& lists, int sta, int en)
    {
        if (sta >= en)
        {
            return;
        }
        int mid = (sta + en) >> 1;
        Do_mergeKLists(lists, sta, mid);
        Do_mergeKLists(lists, mid + 1, en);
        if (mid + 1 <= en)
        {
            ListNode *head = 0, *node = 0, *head1 = lists[sta], *head2 = lists[mid + 1];
            while (head1 != 0 || head2 != 0)
            {
                if ((head2 == 0 && head1 != 0) || (head1 != 0 && head1->val <= head2->val))
                {
                    if (head == 0)
                    {
                        head = head1;
                    }
                    else
                    {
                        node->next = head1;
                    }
                    node = head1;
                    head1 = head1->next;
                }
                else if ((head1 == 0 && head2 != 0) || (head2 != 0 && head2->val <= head1->val))
                {
                    if (head == 0)
                    {
                        head = head2;
                    }
                    else
                    {
                        node->next = head2;
                    }
                    node = head2;
                    head2 = head2->next;
                }
            }
            if (node != 0)
            {
                node->next = 0;
            }
            lists[sta] = head;
        }
    }


    // 10
    // 正则表达式
    /*
        给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
        '.' 匹配任意单个字符
        '*' 匹配零个或多个前面的那一个元素
        所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
        说明:
            s 可能为空，且只包含从 a-z 的小写字母。
            p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
    */
    // 提示：如果没有*，直接比较；如果有*，那么肯定在第二位，如果第一位匹配，那么可以i+1,j；如果第一位不匹配，那么可以i,j+2。
    // 记忆化搜索：因为前一个Do_isMatch会做很多，后一个Do_isMatch会重复
    // 注意：s="abc",p="abcd"为false
    // 注意：不能if (i >= s.size())  ans = (j >= p.size()); 因为s= "aaa",p="a*"，实际上可能j一直在0.
    vector<vector<int> > Dp_isMatch;
    bool isMatch(string s, string p)
    {
        Dp_isMatch.resize(s.size() + 1);
        for (int i = 0; i <= s.size(); i++)
        {
            Dp_isMatch[i].resize(p.size() + 1, -1);
        }
        return Do_isMatch(0, 0, s, p);
    }
    bool Do_isMatch(int i, int j, string s, string p)
    {
        if (Dp_isMatch[i][j] != -1)
        {
            return Dp_isMatch[i][j] == 1;
        }
        bool ans = false;
        if (j >= p.size())
        {
            ans = (i >= s.size());
        }
        else
        {
            bool firstMatch = i < s.size() && (s[i] == p[j] || p[j] == '.');
            if (j + 1 < p.size() && p[j + 1] == '*')
            {
                ans = Do_isMatch(i, j + 2, s, p) || (firstMatch && Do_isMatch(i + 1, j, s, p));
            }
            else
            {
                ans = firstMatch && Do_isMatch(i + 1, j + 1, s, p);
            }
        }
        Dp_isMatch[i][j] = ans ? 1 : 0;
        return ans;
    }


    // 4
    // 寻找两个有序数组的中位数
    /*
        给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
        请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
        你可以假设 nums1 和 nums2 不会同时为空。
    */
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
    {
        int len1 = nums1.size(), len2 = nums2.size();
        if (len1 == 0)
        {
            if (len2 & 1)
            {
                return nums2[len2 >> 1] * 1.0;
            }
            else
            {
                return (nums2[(len2 >> 1) - 1] + nums2[len2 >> 1]) * 0.5;
            }
        }
        if (len2 == 0)
        {
            if (len1 & 1)
            {
                return nums1[len1 >> 1] * 1.0;
            }
            else
            {
                return (nums1[(len1 >> 1) - 1] + nums1[len1 >> 1]) * 0.5;
            }
        }
        if ((len1 + len2) & 1)
        {
            int ans = -1;
            if (!Do_findMedianSortedArrays(nums1, nums2, ((len1 + len2) >> 1) + 1, ans))
            {
                Do_findMedianSortedArrays(nums2, nums1, ((len1 + len2) >> 1) + 1, ans);
            }
            return ans * 1.0;
        }
        else
        {
            int ans1 = -1, ans2 = -1;
            if (!Do_findMedianSortedArrays(nums1, nums2, (len1 + len2) >> 1, ans1))
            {
                Do_findMedianSortedArrays(nums2, nums1, (len1 + len2) >> 1, ans1);
            }
            if (!Do_findMedianSortedArrays(nums1, nums2, ((len1 + len2) >> 1) + 1, ans2))
            {
                Do_findMedianSortedArrays(nums2, nums1, ((len1 + len2) >> 1) + 1, ans2);
            }
            return (ans1 + ans2) * 0.5;
        }
    }
    bool Do_findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2, int index, int &ans)
    {
        int len1 = nums1.size(), len2 = nums2.size();
        int L = 0, R = len1 - 1;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            int need = index - mid - 1;
            if (need < 0)
            {
                R = mid - 1;
            }
            else if (need > len2)
            {
                L = mid + 1;
            }
            else if (need == 0)
            {
                if (nums1[mid] <= nums2[0])
                {
                    ans = nums1[mid];
                    return true;
                }
                L = mid + 1;
            }
            else if (need == len2)
            {
                if (nums1[mid] >= nums2[len2 - 1])
                {
                    ans = nums1[mid];
                    return true;
                }
                R = mid - 1;
            }
            else if (nums1[mid] >= nums2[need - 1] && nums1[mid] <= nums2[need])
            {
                ans = nums1[mid];
                return true;
            }
            else if (nums1[mid] < nums2[need - 1])
            {
                L = mid + 1;
            }
            else
            {
                R = mid - 1;
            }
        }
        return false;
    }
};
int main()
{
    Solution* solution = new Solution();
    return 0;
}
