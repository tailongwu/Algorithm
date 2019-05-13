#include<map>
#include<queue>
#include<stack>
#include<cstdio>
#include<cstring>
#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
 };
class Solution
{
public:
    // 930
    // 和相同的二元子数组
    /*
        在由若干 0 和 1  组成的数组 A 中，有多少个和为 S 的非空子数组。
    */
    int numSubarraysWithSum(vector<int>& A, int S)
    {
        int len = A.size();

    }


    // 102
    // 二叉树的层次遍历
    /*
        给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。
    */
    vector<vector<int> > levelOrder(TreeNode* root)
    {
        vector<vector<int> > ans;
        DFS_levelOrder(root, ans, 0);
        return ans;
    }
    void DFS_levelOrder(TreeNode *node, vector<vector<int> > &ans, int step)
    {
        if (node == 0)
        {
            return;
        }
        if (ans.size() == step)
        {
            vector<int> result;
            ans.push_back(result);
        }
        ans[step].push_back(node->val);
        DFS_levelOrder(node->left, ans, step + 1);
        DFS_levelOrder(node->right, ans, step + 1);
    }


    // 144
    // 二叉树的前序遍历
    /*
        给定一个二叉树，返回它的前序 遍历。
    */
    vector<int> preorderTraversal(TreeNode* root)
    {
        vector<int> ans;
        DFS_preorderTraversal(root, ans);
        return ans;
    }
    void DFS_preorderTraversal(TreeNode *root, vector<int> &ans)
    {
        if (root == 0)
        {
            return;
        }
        ans.push_back(root->val);
        DFS_preorderTraversal(root->left, ans);
        DFS_preorderTraversal(root->right, ans);
    }


    // 94
    // 二叉树的中序遍历
    /*
        给定一个二叉树，返回它的中序 遍历。
    */
    vector<int> inorderTraversal(TreeNode* root)
    {
        vector<int> ans;
        DFS_inorderTraversal(root, ans);
        return ans;
    }
    void DFS_inorderTraversal(TreeNode *root, vector<int> &ans)
    {
        if (root == 0)
        {
            return;
        }
        DFS_inorderTraversal(root->left, ans);
        ans.push_back(root->val);
        DFS_inorderTraversal(root->right, ans);
    }


    // 714
    // 买卖股票的最佳时机含手续费
    /*
        给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。
        你可以无限次地完成交易，但是你每次交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
        返回获得利润的最大值。
    */
    int maxProfit(vector<int>& prices, int fee)
    {
        int len = prices.size();
        if (len == 0)
        {
            return 0;
        }
        vector<int> buy(len);
        vector<int> sell(len);
        buy[0] = -prices[0];
        sell[0] = 0;
        for (int i = 1; i < len; i++)
        {
            buy[i] = max(buy[i - 1], sell[i - 1] - prices[i]);
            sell[i] = max(sell[i - 1], buy[i - 1] + prices[i] - fee);
        }
        return sell[len - 1];
    }


    // 309
    // 最佳买卖股票时机含冷冻期
    /*
        给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
        设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
        你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    */
    // 提示：每种抉择都是这次不行动和这次行动的最大值
    int maxProfit(vector<int>& prices)
    {
        int len = prices.size();
        if (len == 0)
        {
            return 0;
        }
        vector<int> buy(len);
        vector<int> sell(len);
        vector<int> cold(len);
        buy[0] = -prices[0];
        sell[0] = 0;
        cold[0] = 0;
        for (int i = 1; i < len; i++)
        {
            buy[i] = max(cold[i - 1] - prices[i], buy[i - 1]);
            sell[i] = max(buy[i - 1] + prices[i], sell[i - 1]);
            cold[i] = max(sell[i - 1], max(buy[i - 1], cold[i - 1]));
        }
        return sell[len - 1];
    }


    // 1014
    // 最佳观光组合
    /*
        给定正整数数组 A，A[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的距离为 j - i。
        一对景点（i < j）组成的观光组合的得分为（A[i] + A[j] + i - j）：景点的评分之和减去它们两者之间的距离。
        返回一对观光景点能取得的最高分。
    */
    // 提示：（i<j）A[i] + i + A[j] - j最大值，保存A[i] + i的最大值再遍历
    int maxScoreSightseeingPair(vector<int>& A)
    {
        int len = A.size();
        int ans = 0;
        int ma = A[0];
        for (int i = 1; i < len; i++)
        {
            ans = max(ans, A[i] - i + ma);
            if (A[i] + i > ma)
            {
                ma = A[i] + i;
            }
        }
        return ans;
    }


    // 932
    // 漂亮的数组
    /*
        对于某些固定的 N，如果数组 A 是整数 1, 2, ..., N 组成的排列，使得：
        对于每个 i < j，都不存在 k 满足 i < k < j 使得 A[k] * 2 = A[i] + A[j]。
        那么数组 A 是漂亮数组。
        给定 N，返回任意漂亮数组 A（保证存在一个）
    */
    // 提示：奇数放在左边，偶数放在右边。如果奇数和偶数数组都是漂亮数组，那么组合起来也是漂亮数组
    // 如果A数组为漂亮数组，那么A*2数组也为漂亮数组，那么A*2-1数组也是漂亮数组
    // [1] -> [1, 2] -> [1, 3, 2, 4] -> [1, 5, 3, 7, 2, 6, 4, 8]
    vector<int> beautifulArray(int N)
    {
        vector<int> ans(N);
        ans[0] = 1;
        for (int i = 1; ; i *= 2)
        {
            int index = 0;
            for (int j = 0; j < i; j++)
            {
                if (ans[j] * 2 - 1 <= N)
                {
                    ans[index++] = ans[j] * 2 - 1;
                    if (index == N)
                    {
                        break;
                    }
                }
            }
            if (index == N)
            {
                break;
            }
            for (int j = 0; j < i; j++)
            {
                if (ans[j] + 1 <= N)
                {
                    ans[index++] = ans[j] + 1;
                    if (index == N)
                    {
                        break;
                    }
                }
            }
            if (index == N)
            {
                break;
            }
        }
        return ans;
    }


    // 667
    // 优美的排列II
    /*
        给定两个整数 n 和 k，你需要实现一个数组，这个数组包含从 1 到 n 的 n 个不同整数，同时满足以下条件：
        ① 如果这个数组是 [a1, a2, a3, ... , an] ，那么数组 [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] 中应该有且仅有 k 个不同整数；.
        ② 如果存在多种答案，你只需实现并返回其中任意一种.
    */
    vector<int> constructArray(int n, int k)
    {
        vector<int> ans(n);
        vector<bool> vis(n, false);
        ans[0] = 1;
        vis[1] = true;
        int index = 1;
        for (int i = k; i >= 1; i--)
        {
            int ma = ans[index - 1] + i;
            int mi = ans[index - 1] - i;
            if (mi < 1 || vis[mi])
            {
                ans[index] = ma;
                vis[ma] = true;
            }
            else
            {
                ans[index] = mi;
                vis[mi] = true;
            }
            index++;
        }
        for (int i = k + 2; i <= n; i++)
        {
            ans[index++] = i;
        }
        return ans;
    }


    // 526
    // 优美的排列
    /*
        假设有从 1 到 N 的 N 个整数，如果从这 N 个数字中成功构造出一个数组，使得数组的第 i 位 (1 <= i <= N) 满足如下两个条件中的一个，我们就称这个数组为一个优美的排列。条件：
        第 i 位的数字能被 i 整除
        i 能被第 i 位上的数字整除
        现在给定一个整数 N，请问可以构造多少个优美的排列？
    */
    int countArrangement(int N)
    {
        vector<bool> vis(N + 1, false);
        int ans = 0;
        DFS_countArrangement(N, ans, vis, 1);
        return ans;
    }
    void DFS_countArrangement(int N, int &ans, vector<bool> &vis, int stp)
    {
        if (stp == N + 1)
        {
            ans++;
            return;
        }
        for (int i = 1; i <= N; i++)
        {
            if (!vis[i] && (i % stp == 0 || stp % i == 0))
            {
                vis[i] = true;
                DFS_countArrangement(N, ans, vis, stp + 1);
                vis[i] = false;
            }
        }
    }


    // 419
    // 甲板上的战舰
    /*
        给定一个二维的甲板， 请计算其中有多少艘战舰。 战舰用 'X'表示，空位用 '.'表示。 你需要遵守以下规则：
        给你一个有效的甲板，仅由战舰或者空位组成。
        战舰只能水平或者垂直放置。换句话说,战舰只能由 1xN (1 行, N 列)组成，或者 Nx1 (N 行, 1 列)组成，其中N可以是任意大小。
        两艘战舰之间至少有一个水平或垂直的空位分隔 - 即没有相邻的战舰。
        进阶:你可以用一次扫描算法，只使用O(1)额外空间，并且不修改甲板的值来解决这个问题吗？
    */
    int countBattleships(vector<vector<char> >& board)
    {
        int row = board.size();
        int col = board[0].size();
        int ans = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (board[i][j] == 'X')
                {
                    if (i > 0 && board[i - 1][j] == 'X')
                    {
                        continue;
                    }
                    if (j > 0 && board[i][j - 1] == 'X')
                    {
                        continue;
                    }
                    ans++;
                }
            }
        }
        return ans;
    }


    // 227
    // 基本计算器II
    /*
        实现一个基本的计算器来计算一个简单的字符串表达式的值。
        字符串表达式仅包含非负整数，+， - ，*，/ 四种运算符和空格  。 整数除法仅保留整数部分。
    */
    int calculate(string s)
    {
        int len = s.size(), ans = 0;
        stack<char> ops;
        stack<int> num;
        for (int i = 0; i < len; i++)
        {
            if (s[i] == '+' || s[i] == '-')
            {
                while (ops.size() > 0)
                {
                    char op = ops.top();
                    ops.pop();
                    int num2 = num.top();
                    num.pop();
                    int num1 = num.top();
                    num.pop();
                    num.push(do_calculate(num1, num2, op));
                }
                ops.push(s[i]);
            }
            else if (s[i] == '*' || s[i] == '/')
            {
                if (ops.size() > 0 && (ops.top() == '*' || ops.top() == '/'))
                {
                    char op = ops.top();
                    ops.pop();
                    int num2 = num.top();
                    num.pop();
                    int num1 = num.top();
                    num.pop();
                    num.push(do_calculate(num1, num2, op));
                }
                ops.push(s[i]);
            }
            else if (s[i] >= '0' && s[i] <= '9')
            {
                long long k = 0;
                int j = i;
                while (j < len && s[j] >= '0' && s[j] <= '9')
                {
                    k = k * 10 + s[j] - '0';
                    j++;
                }
                i = j - 1;
                num.push(k);
            }
        }
        while (!ops.empty())
        {
            char op = ops.top();
            ops.pop();
            int num2 = num.top();
            num.pop();
            int num1 = num.top();
            num.pop();
            num.push(do_calculate(num1, num2, op));
        }
        return num.top();
    }
    int do_calculate(int num1, int num2, char op)
    {
        switch(op)
        {
            case '+': return num1 + num2;
            case '-': return num1 - num2;
            case '*': return num1 * num2;
            case '/': return num1 / num2;
        }
        return 0;
    }


    // 948
    // 令牌放置
    /*
        你的初始能量为 P，初始分数为 0，只有一包令牌。
        令牌的值为 token[i]，每个令牌最多只能使用一次，可能的两种使用方法如下：
        如果你至少有 token[i] 点能量，可以将令牌置为正面朝上，失去 token[i] 点能量，并得到 1 分。
        如果我们至少有 1 分，可以将令牌置为反面朝上，获得 token[i] 点能量，并失去 1 分。
        在使用任意数量的令牌后，返回我们可以得到的最大分数。
    */
    int bagOfTokensScore(vector<int>& tokens, int P)
    {
        sort (tokens.begin(), tokens.end());
        int L = 0, R = tokens.size() - 1, ans = 0;
        while (L <= R)
        {
            if (P >= tokens[L])
            {
                ans++;
                P -= tokens[L];
                L++;
            }
            else
            {
                if (ans > 0)
                {
                    P += (tokens[R] - tokens[L]);
                    L++;
                    R--;
                }
                else
                {
                    return ans;
                }
            }
        }
        return ans;
    }


    // 735
    // 星星碰撞
    /*
        给定一个整数数组 asteroids，表示在同一行的行星。
        对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。
        找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。
    */
    vector<int> asteroidCollision(vector<int>& asteroids)
    {
        vector<int> ans;
        int len = asteroids.size();
        for (int i = 0; i < len; i++)
        {
            if (asteroids[i] > 0)
            {
                ans.push_back(asteroids[i]);
            }
            else
            {
                bool push = true;
                while (ans.size() > 0)
                {
                    if (ans.back() < 0)
                    {
                        ans.push_back(asteroids[i]);
                        push = false;
                        break;
                    }
                    else if (ans.back() > -asteroids[i])
                    {
                        push = false;
                        break;
                    }
                    else if (ans.back() < -asteroids[i])
                    {
                        ans.pop_back();
                    }
                    else if(ans.back() == -asteroids[i])
                    {
                        ans.pop_back();
                        push = false;
                        break;
                    }
                }
                if (push)
                {
                    ans.push_back(asteroids[i]);
                }
            }
        }
        return ans;
    }


    // 954
    // 二倍数对数组
    /*
        给定一个长度为偶数的整数数组 A，只有对 A 进行重组后可以满足 “对于每个 0 <= i < len(A) / 2，都有 A[2 * i + 1] = 2 * A[2 * i]” 时，返回 true；否则，返回 false。
    */
    bool canReorderDoubled(vector<int>& A)
    {
        int len = A.size();
        vector<int> p;
        vector<int> n;
        map<int, int> m;
        for (int i = 0; i < len; i++)
        {
            m[A[i]]++;
            if (A[i] >= 0)
            {
                p.push_back(A[i]);
            }
            else
            {
                n.push_back(A[i]);
            }
        }
        if (p.size() %2 != 0)
        {
            return false;
        }
        sort(p.begin(), p.end());
        sort(n.begin(), n.end());
        int len1 = p.size(), len2 = n.size();
        for (int i = 0; i < len1; i++)
        {
            if (m[p[i]] > 0)
            {
                m[p[i]]--;
                if (m[p[i] * 2] == 0)
                {
                    return false;
                }
                else
                {
                    m[p[i] * 2]--;
                }
            }
        }
        for (int i = len2 - 1; i >= 0; i--)
        {
            if (m[n[i]] > 0)
            {
                m[n[i]]--;
                if (m[n[i] * 2] == 0)
                {
                    return false;
                }
                else
                {
                    m[n[i] * 2]--;
                }
            }
        }
        return true;
    }


    // 565
    // 数组嵌套
    /*
        索引从0开始长度为N的数组A，包含0到N - 1的所有整数。找到并返回最大的集合S，S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }且遵守以下的规则。
        假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素
    */
    int arrayNesting(vector<int>& nums)
    {
        int len = nums.size(), ma = 0;
        vector<bool> vis(len, false);
        for (int i = 0; i < len; i++)
        {
            if (vis[i])
            {
                continue;
            }
            int j = i, cnt = 0;
            while (!vis[j])
            {
                vis[j] = true;
                j = nums[j];
                cnt++;
            }
            if (cnt > ma)
            {
                ma = cnt;
            }
        }
        return ma;
    }


    // 413
    // 等差数列划分
    /*
        数组 A 包含 N 个数，且索引从0开始。数组 A 的一个子数组划分为数组 (P, Q)，P 与 Q 是整数且满足 0<=P<Q<N 。
        如果满足以下条件，则称子数组(P, Q)为等差数组：
        元素 A[P], A[p + 1], ..., A[Q - 1], A[Q] 是等差的。并且 P + 1 < Q 。
        函数要返回数组 A 中所有为等差数组的子数组个数。
    */
    int numberOfArithmeticSlices(vector<int>& A)
    {
        int ans = 0;
        int len = A.size();
        for (int i = 0; i < len - 2; i++)
        {
            int d = A[i + 1] - A[i];
            int j = i + 2;
            while (j < len && A[j] - A[j - 1] == d)
            {
                j++;
            }
            ans = ans + (j - i - 2 + 1) * (j - i - 2) / 2;
            i = j - 2;
        }
        return ans;
    }


    // 797
    // 所有可能路径
    /*
        给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）
        二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了a→b你就不能从b→a）空就是没有下一个结点了。
    */
    vector<vector<int> > allPathsSourceTarget(vector<vector<int> >& graph)
    {
        vector<vector<int> > ans;
        vector<int> result;
        result.push_back(0);
        DFS_allPathsSourceTarget(graph, ans, result, 0);
        return ans;
    }
    void DFS_allPathsSourceTarget(vector<vector<int> > &graph, vector<vector<int> > &ans, vector<int> &result, int k)
    {
        if (k == graph.size() - 1)
        {
            ans.push_back(result);
        }
        for (int i = 0; i < graph[k].size(); i++)
        {
            if (graph[k][i] > k)
            {
                result.push_back(graph[k][i]);
                DFS_allPathsSourceTarget(graph, ans, result, graph[k][i]);
                result.pop_back();
            }
        }
    }


    // 863
    // 二叉树中所有距离为 K 的结点
    /*
        给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。
        返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。
    */
    // 提示：先找出到目标点的path，然后遍历每个节点的另一个儿子，注意当前节点也可能是答案
    vector<int> distanceK(TreeNode* root, TreeNode* target, int K)
    {
        vector<int> ans;
        vector<TreeNode*> path;
        FindPath_distanceK(root, target, path);
        int len = path.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *node;
            int D;
            if (i != len - 1)
            {
                if (path[i + 1] != path[i]->left)
                {
                    node = path[i]->left;
                }
                else
                {
                    node = path[i]->right;
                }
                D = K - (len - i);
            }
            else
            {
                node = path[i];
                D = K;
            }
            if (D >= 0)
            {
                FindAns_distanceK(node, ans, D, 0);
            }
            else if (D == -1)
            {
                ans.push_back(path[i]->val);
            }
        }
        return ans;
    }
    void FindPath_distanceK(TreeNode *node, TreeNode *target, vector<TreeNode*> &path)
    {
        if (node == 0 || (path.size() > 0 && path.back() == target))
        {
            return;
        }
        path.push_back(node);
        if (node == target)
        {
            return;
        }
        FindPath_distanceK(node->left, target, path);
        if (path.size() > 0 && path.back() == target)
        {
            return;
        }
        FindPath_distanceK(node->right, target, path);
        if (path.size() > 0 && path.back() == target)
        {
            return;
        }
        path.pop_back();
    }
    void FindAns_distanceK(TreeNode *node, vector<int> &ans, int K, int D)
    {
        if (node == 0)
        {
            return;
        }
        if (K == D)
        {
            ans.push_back(node->val);
            return;
        }
        FindAns_distanceK(node->left, ans, K, D + 1);
        FindAns_distanceK(node->right, ans, K, D + 1);
    }


    // 870
    // 优势洗牌
    /*
        给定两个大小相等的数组 A 和 B，A 相对于 B 的优势可以用满足 A[i] > B[i] 的索引 i 的数目来描述。
        返回 A 的任意排列，使其相对于 B 的优势最大化。
    */
    vector<int> advantageCount(vector<int>& A, vector<int>& B)
    {
        sort(A.begin(), A.end());
        int len = A.size();
        vector<int> ans(len);
        vector<bool> vis(len);
        for (int i = 0; i < len; i++)
        {
            vis[i] = false;
        }
        for (int i = 0; i < len; i++)
        {
            int L = 0, R = len - 1, mid, ansIndex = 0;
            while (L <= R)
            {
                mid = (L + R) >> 1;
                if (A[mid] > B[i])
                {
                    ansIndex = mid;
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }
            while (true)
            {
                if (!vis[ansIndex])
                {
                    ans[i] = A[ansIndex];
                    vis[ansIndex] = true;
                    break;
                }
                ansIndex++;
                if (ansIndex == len)
                {
                    ansIndex = 0;
                }
            }
        }
        return ans;
    }


    // 306
    // 累加数
    /*
        累加数是一个字符串，组成它的数字可以形成累加序列。
        一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
        给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。
        说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。
        比如：112358，199100199
    */
    bool isAdditiveNumber(string num)
    {
        int len = num.size();
        for (int i = 1; i < len; i++)
        {
            string a = "";
            for (int r = 0; r < i; r++)
            {
                a = a + num[r];
            }
            if (a[0] == '0' && i > 1)
            {
                continue;
            }
            for (int j = i; j < len; j++)
            {
                string b = "";
                for (int r = i; r <= j; r++)
                {
                    b = b + num[r];
                }
                if (b[0] == '0' && j > i)
                {
                    continue;
                }
                if (DFS_isAdditiveNumber(num, len, j + 1, b, Add_isAdditiveNumber(a, b)))
                {
                    return true;
                }
            }
        }
        return false;
    }
    bool DFS_isAdditiveNumber(string num, int len, int startIndex, string last, string sum)
    {
        int sumLen = sum.size();
        if (len - startIndex < sumLen)
        {
            return false;
        }
        int newIndex = startIndex;
        while (newIndex - startIndex < sumLen)
        {
            if (sum[newIndex - startIndex] != num[newIndex])
            {
                return false;
            }
            newIndex++;
        }
        if (startIndex + sumLen == len)
        {
            return true;
        }
        return DFS_isAdditiveNumber(num, len, startIndex + sumLen, sum, Add_isAdditiveNumber(last, sum));
    }
    string Add_isAdditiveNumber(string a, string b)
    {
        int lenA = a.size(), lenB = b.size();
        int L = 0, R = lenA - 1;
        while (L < R)
        {
            swap(a[L++], a[R--]);
        }
        L = 0, R = lenB - 1;
        while (L < R)
        {
            swap(b[L++], b[R--]);
        }
        string ans = "";
        int r = 0, addA, addB, sum;
        for (int i = 0; i < lenA || i < lenB; i++)
        {
            if (i >= lenA)
            {
                addA = 0;
            }
            else
            {
                addA = a[i] - '0';
            }
            if (i >= lenB)
            {
                addB = 0;
            }
            else
            {
                addB = b[i] - '0';
            }
            sum = addA + addB + r;
            r = sum / 10;
            sum %= 10;
            ans = ans + (char)(sum + '0');
        }
        if (r != 0)
        {
            ans = ans + (char)(r + '0');
        }
        L = 0, R = ans.size() - 1;
        while (L < R)
        {
            swap(ans[L++], ans[R--]);
        }
        return ans;
    }


    // 445
    // 两数相加II
    /*
        给定两个非空链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储单个数字。将这两数相加会返回一个新的链表。
        进阶: 如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。
    */
    ListNode* addTwoNumbersII(ListNode* l1, ListNode* l2)
    {
        l1 = Reverse_addTwoNumbers(l1);
        l2 = Reverse_addTwoNumbers(l2);
        int r = 0, a, b, sum;
        ListNode *cur1 = l1, *cur2 = l2, *head = NULL, *cur;
        while (cur1 != NULL || cur2 != NULL)
        {
            if (cur1 == NULL)
            {
                a = 0;
            }
            else
            {
                a = cur1->val;
                cur1 = cur1->next;
            }
            if (cur2 == NULL)
            {
                b = 0;
            }
            else
            {
                b = cur2->val;
                cur2 = cur2->next;
            }
            sum = a + b + r;
            r = sum / 10;
            sum %= 10;
            if (head == NULL)
            {
                head = new ListNode(sum);
                cur = head;
            }
            else
            {
                cur->next = new ListNode(sum);
                cur = cur->next;
            }
        }
        if (r != 0)
        {
            cur->next = new ListNode(r);
            cur = cur->next;
        }
        l1 = Reverse_addTwoNumbers(l1);
        l2 = Reverse_addTwoNumbers(l2);
        head = Reverse_addTwoNumbers(head);
        return head;
    }
    ListNode* Reverse_addTwoNumbers(ListNode *l)
    {
        ListNode *head = NULL, *cur = l, *next;
        while (cur != NULL)
        {
            next = cur->next;
            cur->next = head;
            head = cur;
            cur = next;
        }
        return head;
    }


    // 2
    // 两数相加
    /*
        给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
        如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
        您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
    */
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
    {
        int r = 0, a, b, sum;
        ListNode *cur1 = l1, *cur2 = l2, *head = NULL, *cur;
        while (cur1 != NULL || cur2 != NULL)
        {
            if (cur1 == NULL)
            {
                a = 0;
            }
            else
            {
                a = cur1->val;
                cur1 = cur1->next;
            }
            if (cur2 == NULL)
            {
                b = 0;
            }
            else
            {
                b = cur2->val;
                cur2 = cur2->next;
            }
            sum = a + b + r;
            r = sum / 10;
            sum %= 10;
            if (head == NULL)
            {
                head = new ListNode(sum);
                cur = head;
            }
            else
            {
                cur->next = new ListNode(sum);
                cur = cur->next;
            }
        }
        if (r != 0)
        {
            cur->next = new ListNode(r);
            cur = cur->next;
        }
        return head;
    }


    // 454
    // 四数相加II
    /*
        给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。
        为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -228 到 228 - 1 之间，最终结果不会超过 231 - 1 。
    */
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D)
    {
        int len = A.size();
        map<int, int> m;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                m[A[i] + B[j]]++;
            }
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                int sum = -(C[i] + D[j]);
                if (m[sum] > 0)
                {
                    ans += m[sum];
                }
            }
        }
        return ans;
    }


    // 18
    // 四数之和
    /*
        给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
    */
    // 提示：固定两个数，双指针找另外两个；保存两两之和，再双指针
    vector<vector<int> > fourSum(vector<int>& nums, int target)
    {
        vector<vector<int> > ans;
        int len = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 0; i < len; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1])
            {
                continue;
            }
            for (int j = i + 1; j < len; j++)
            {
                if (j > i + 1 && nums[j] == nums[j - 1])
                {
                    continue;
                }
                int L = j + 1, R = len - 1;
                while (L < R)
                {
                    int sum = nums[i] + nums[j] + nums[L] + nums[R];
                    if (sum < target)
                    {
                        L++;
                    }
                    else if (sum > target)
                    {
                        R--;
                    }
                    else
                    {
                        int ansSize = ans.size();
                        if (!(ansSize > 0
                            && ans[ansSize - 1][0] == nums[i]
                            && ans[ansSize - 1][1] == nums[j]
                            && ans[ansSize - 1][2] == nums[L]
                            && ans[ansSize - 1][3] == nums[R]))
                        {
                            vector<int> result(4);
                            result[0] = nums[i];
                            result[1] = nums[j];
                            result[2] = nums[L];
                            result[3] = nums[R];
                            ans.push_back(result);
                        }
                        L++;
                        R--;
                    }
                }
            }
        }
        return ans;
    }


    // 923
    // 三数之和的多种可能
    /*
        给定一个整数数组 A，以及一个整数 target 作为目标值，返回满足 i < j < k 且 A[i] + A[j] + A[k] == target 的元组 i, j, k 的数量。
        由于结果会非常大，请返回 结果除以 10^9 + 7 的余数。
    */
    // 提示：三个数不同，两个数相同，三个数相同
    int threeSumMulti(vector<int>& A, int target)
    {
        long long ans = 0;
        int len = A.size();
        sort(A.begin(), A.end());
        vector<int> nums(101);
        for (int i = 0; i < 101; i++)
        {
            nums[i] = 0;
        }
        for (int i = 0; i < len; i++)
        {
            nums[A[i]]++;
        }
        int cur = 0;
        for (int i = 0; i < len; )
        {
            int j = i + 1;
            while (j < len && A[j] == A[i])
            {
                j++;
            }
            A[cur++] = A[i];
            i = j;
        }
        for (int i = 0; i < cur; i++)
        {
            int L = i + 1, R = cur - 1;
            while (L < R)
            {
                int sum = A[i] + A[L] + A[R];
                if (sum < target)
                {
                    L++;
                }
                else if (sum > target)
                {
                    R--;
                }
                else
                {
                    ans = (ans + (long long)nums[A[i]] * nums[A[L]] * nums[A[R]]) % 1000000007;
                    L++;
                    R--;
                }
            }
        }
        for (int i = 0; i < 101; i++)
        {
            if (nums[i] >= 2)
            {
                int k = target - i * 2;
                if (k >= 0 && k < 101 && k != i && nums[k] > 0)
                {
                    ans = (ans + (long long)nums[i] * (nums[i] - 1) / 2 * nums[k]) % 1000000007;
                }
            }
        }
        if (target % 3 == 0)
        {
            int k = target / 3;
            if (k >= 0 && k < 101 && nums[k] >= 3)
            {
                ans = (ans + (long long)nums[k] * (nums[k] - 1) * (nums[k] - 2) / 6) % 1000000007;
            }
        }
        return (int)ans;
    }


    // 16
    // 最接近的三数之和
    /*
        给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
    */
    int threeSumClosest(vector<int>& nums, int target)
    {
        int len = nums.size();
        int mi = -1, ans;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < len; i++)
        {
            int L = i + 1, R = len - 1;
            while (L < R)
            {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum < target)
                {
                    L++;
                    if (mi == -1 || target - sum < mi)
                    {
                        mi = target - sum;
                        ans = sum;
                    }
                }
                else if (sum > target)
                {
                    R--;
                    if (mi == -1 || sum - target < mi)
                    {
                        mi = sum - target;
                        ans = sum;
                    }
                }
                else
                {
                    return sum;
                }
            }
        }
        return ans;
    }


    // 15
    // 三数之和
    /*
        给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
    */
    vector<vector<int> > threeSum(vector<int>& nums)
    {
        vector<vector<int> > ans;
        int len = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 0; i < len; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1])
            {
                continue;
            }
            int sum = -nums[i];
            int L = i + 1, R = len - 1;
            while (L < R)
            {
                int s = nums[L] + nums[R];
                if (L == i || s < sum)
                {
                    L++;
                    continue;
                }
                if (R == i || s > sum)
                {
                    R--;
                    continue;
                }
                if (s == sum)
                {
                    int ansSize = ans.size();
                    if (!(ansSize > 0
                        && nums[i] == ans[ansSize - 1][0]
                        && nums[L] == ans[ansSize - 1][1]
                        && nums[R] == ans[ansSize - 1][2]))
                    {
                        vector<int> result(3);
                        result[0] = nums[i];
                        result[1] = nums[L];
                        result[2] = nums[R];
                        ans.push_back(result);
                    }
                    L++;
                    R--;
                }
            }
        }
        return ans;
    }

    // 650
    // 只有两个键的键盘
    /*
        最初在一个记事本上只有一个字符 'A'。你每次可以对这个记事本进行两种操作：
        Copy All (复制全部) : 你可以复制这个记事本中的所有字符(部分的复制是不允许的)。
        Paste (粘贴) : 你可以粘贴你上一次复制的字符。
        给定一个数字 n 。你需要使用最少的操作次数，在记事本中打印出恰好 n 个 'A'。输出能够打印出 n 个 'A' 的最少操作次数。
    */
    int minSteps(int n)
    {
        // 分析比如30，如果分解成1*30=1+30；2*15=2+15；3*10=3+10；5*6=5+6；2*3*5=2+3+5；
        int ans = 0;
        if (n < 2)
        {
            return 0;
        }
        for (int i = 2; i <= n; i++)
        {
            if (n % i == 0)
            {
                while (n % i == 0)
                {
                    ans += i;
                    n /= i;
                }
            }
        }
        return ans;
    }
//    int minSteps(int n)
//    {
//        vector<int> dp(n + 1);
//        for (int i = 0; i <= n; i++)
//        {
//            dp[i] = 1e9;
//        }
//        dp[1] = 0;
//        for (int i = 2; i <= n; i++)
//        {
//            for (int j = 2; j <= i; j++)
//            {
//                if (i % j == 0)
//                {
//                    dp[i] = min(dp[i], dp[i / j] + i / j);
//                }
//            }
//        }
//        return dp[n];
//    }
//    struct Q_minSteps
//    {
//        int step, len, copyLen;
//    };
//    int minSteps(int n)
//    {
//        queue<Q_minSteps> Q;
//        Q_minSteps p, q;
//        p.step = 0;
//        p.len = 1;
//        p.copyLen = 0;
//        Q.push(p);
//        while (!Q.empty())
//        {
//            p = Q.front();
//            Q.pop();
//            if (p.len == n)
//            {
//                while (!Q.empty())
//                {
//                    Q.pop();
//                }
//                return p.step;
//            }
//            if (p.copyLen != 0)
//            {
//                q.len = p.len + p.copyLen;
//                q.step = p.step + 1;
//                q.copyLen = p.copyLen;
//                if (q.len <= n)
//                {
//                    Q.push(q);
//                }
//            }
//            if (p.len != p.copyLen)
//            {
//                q.len = p.len;
//                q.step = p.step + 1;
//                q.copyLen = p.len;
//                Q.push(q);
//            }
//        }
//        return 0;
//    }


    // 456
    // 132模式
    /*
        给定一个整数序列：a1, a2, ..., an，一个132模式的子序列 ai, aj, ak 被定义为：当 i < j < k 时，ai < ak < aj。设计一个算法，当给定有 n 个数字的序列时，验证这个序列中是否含有132模式的子序列。
    */
    bool find132pattern(vector<int>& nums)
    {
        int len = nums.size();
        if (len == 0)
        {
            return false;
        }

        vector<int> mi(len);
        mi[0] = nums[0];
        for (int i = 1; i < len; i++)
        {
            mi[i] = mi[i - 1] < nums[i] ? mi[i - 1] : nums[i];
        }
        stack<int> sta;
        int ma = 1e9;
        for (int i = len - 1; i > 0; i--)
        {
            if (nums[i] > mi[i])
            {
                while (!sta.empty() && sta.top() < nums[i])
                {
                    ma = sta.top();
                    sta.pop();
                }
                if (ma < nums[i] && ma > mi[i] < ma)
                {
                    return true;
                }
            }
            sta.push(nums[i]);
        }
        return false;
    }


    // 542
    // 01矩阵
    /*
        给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
        两个相邻元素间的距离为 1
    */
    vector<vector<int> > updateMatrix(vector<vector<int> >& matrix)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        vector<vector<int> > ans(row);
        for (int i = 0; i < row; i++)
        {
            ans[i].resize(col);
        }
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (matrix[i][j] == 0)
                {
                    ans[i][j] = 0;
                    continue;
                }
                bool getAns = false;
                for (int k = 1; !getAns; k++)
                {
                    for (int x = 0; x <= k && !getAns; x++)
                    {
                        int y = k - x;
                        int xx = i + x;
                        int yy = j + y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i - x;
                        yy = j + y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i - x;
                        yy = j - y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i + x;
                        yy = j - y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                    }
                }
            }
        }
        return ans;
    }


    // 890
    // 查找和替换模式
    /*
        你有一个单词列表 words 和一个模式  pattern，你想知道 words 中的哪些单词与模式匹配。
        如果存在字母的排列 p ，使得将模式中的每个字母 x 替换为 p(x) 之后，我们就得到了所需的单词，那么单词与模式是匹配的。
        （回想一下，字母的排列是从字母到字母的双射：每个字母映射到另一个字母，没有两个字母映射到同一个字母。）
        返回 words 中与给定模式匹配的单词列表。
        你可以按任何顺序返回答案。
    */
    vector<string> findAndReplacePattern(vector<string>& words, string pattern)
    {
        vector<string> ans;
        int si = words.size();
        int len = pattern.size();
        for (int i = 0; i < si; i++)
        {
            int len1 = words[i].size();
            if (len1 != len)
            {
                continue;
            }
            map<char, char> m;
            for (int j = 0; j < len1; j++)
            {
                m[words[i][j]] = 0;
            }
            bool yes = true;
            for (int j = 0; j < len1; j++)
            {
                if (!m[words[i][j]])
                {
                    m[words[i][j]] = pattern[j];
                }
                else if (m[words[i][j]] != pattern[j])
                {
                    yes = false;
                    break;
                }
            }
            if (yes)
            {
                ans.push_back(words[i]);
            }
        }
        return ans;
    }


    // 216
    // 组合总和III
    /*
        找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
    */
    bool v_combinationSum3[10];
    vector<int> a_combinationSum3;
    vector<vector<int> > combinationSum3(int k, int n)
    {
        memset(v_combinationSum3, false, sizeof(v_combinationSum3));
        a_combinationSum3.resize(10);
        vector<vector<int> > ans;
        do_combinationSum3(ans, n, k, 0, 0);
        return ans;
    }
    void do_combinationSum3(vector<vector<int> > &ans, int n, int k, int stp, int sum)
    {
        if (sum > n || k > 9)
        {
            return;
        }
        if (k == stp)
        {
            if (sum == n)
            {
                vector<int> result(k);
                for(int i = 0; i < k; i++)
                {
                    result[i] = a_combinationSum3[i];
                }
                ans.push_back(result);
            }
            return;
        }
        for (int i = stp == 0 ? 0 : a_combinationSum3[stp - 1]; i < 9; i++)
        {
            if (!v_combinationSum3[i])
            {
                v_combinationSum3[i] = true;
                a_combinationSum3[stp] = i + 1;
                do_combinationSum3(ans, n, k, stp + 1, sum + i + 1);
                v_combinationSum3[i] = false;
            }
        }
    }

    // 77
    // 组合
    /*
        给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
    */
    vector<bool> v_combine;
    vector<int> a_combine;
    vector<vector<int> > combine(int n, int k)
    {
        v_combine.resize(n);
        a_combine.resize(k);
        for (int i = 0; i < n; i++)
        {
            v_combine[i] = false;
        }
        vector<vector<int> > ans;
        do_combine(ans, n, k, 0);
        return ans;
    }
    void do_combine(vector<vector<int> > &ans, int n, int k, int stp)
    {
        if (k == stp)
        {
            vector<int> result(k);
            for (int i = 0; i < k; i++)
            {
                result[i] = a_combine[i];
            }
            ans.push_back(result);
            return;
        }
        for (int i = stp == 0 ? 0 : a_combine[stp - 1]; i < n; i++)
        {
            if (!v_combine[i])
            {
                v_combine[i] = true;
                a_combine[stp] = i + 1;
                do_combine(ans, n, k, stp + 1);
                v_combine[i] = false;
            }
        }
    }

    // 894
    // 所有可能的满二叉树
    /*
        满二叉树是一类二叉树，其中每个结点恰好有 0 或 2 个子结点。
        返回包含 N 个结点的所有可能满二叉树的列表。 答案的每个元素都是一个可能树的根结点。
        答案中每个树的每个结点都必须有 node.val=0。
        你可以按任何顺序返回树的最终列表。
    */
    vector<TreeNode*> allPossibleFBT(int N)
    {
        vector<TreeNode*> ans;
        if (N & 1)
        {
            do_allPossibleFBT(N, ans, 0, 0, 0);
        }
        return ans;
    }
    void do_allPossibleFBT(int N, vector<TreeNode*> ans, int step, TreeNode *node, TreeNode *root)
    {
        if (N < step)
        {
            return;
        }
        if (N == step)
        {
            ans.push_back(root);
            return;
        }
        if (node == 0)
        {
            node = new TreeNode(0);
            node->left = 0;
            node->right = 0;
            do_allPossibleFBT(N, ans, step + 1, node, node);
            return;
        }

        TreeNode *left = new TreeNode(0);
        left->left = 0;
        left->right = 0;
        TreeNode *right = new TreeNode(0);
        right->left = 0;
        right->right = 0;
        node->left = left;
        node->right = right;

        do_allPossibleFBT(N, ans, step + 2, left, root);
        do_allPossibleFBT(N, ans, step + 2, right, root);

        node->left = 0;
        node->right = 0;
    }


    // 46
    // 全排列
    bool v_permute[20];
    int result_permute[20];
    vector<vector<int> > permute(vector<int>& nums)
    {
        memset(v_permute, false, sizeof(v_permute));
        vector<vector<int> > ans;
        do_permute(nums, nums.size(), 0, ans);
        return ans;
    }
    void do_permute(vector<int>& nums, int length, int step, vector<vector<int> >& ans)
    {
        if (step == length)
        {
            vector<int> result(step);
            for (int i = 0; i < step; i++)
            {
                result[i] = result_permute[i];
            }
            ans.push_back(result);
            return;
        }
        for (int i = 0; i < length; i++)
        {
            if (!v_permute[i])
            {
                v_permute[i] = true;
                result_permute[step] = nums[i];
                do_permute(nums, length, step + 1, ans);
                v_permute[i] = false;
            }
        }
    }


    // 814
    // 二叉树剪枝
    /*
        给定二叉树根结点 root ，此外树的每个结点的值要么是 0，要么是 1。
        返回移除了所有不包含 1 的子树的原二叉树。
    */
    TreeNode* pruneTree(TreeNode* root)
    {
        do_pruneTree(root);
        return root;
    }
    bool do_pruneTree(TreeNode* node)
    {
        if (node == 0)
        {
            return false;
        }
        bool left = do_pruneTree(node->left);
        bool right = do_pruneTree(node->right);
        if (!left)
        {
            node->left = 0;
        }
        if (!right)
        {
            node->right = 0;
        }
        if (node->val == 1 || left || right)
        {
            return true;
        }
        return false;
    }


    // 861
    // 翻转矩阵后的得分
    /*
        有一个二维矩阵 A 其中每个元素的值为 0 或 1 。
        移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 0 都更改为 1，将所有 1 都更改为 0。
        在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。
        返回尽可能高的分数。
    */
    int matrixScore(vector<vector<int> >& A)
    {
        int f[25];
        f[0] = 1;
        for (int i = 1; i < 25; i++)
        {
            f[i] = f[i - 1] << 1;
        }
        int row = A.size();
        int col = A[0].size();
        for (int i = 0; i < row; i++)
        {
            if (A[i][0] == 0)
            {
                for (int j = 0; j < col; j++)
                {
                    A[i][j] = 1 - A[i][j];
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < col; i++)
        {
            int cnt = 0;
            for (int j = 0; j < row; j++)
            {
                cnt += A[j][i];
            }
            if (cnt < row - cnt)
            {
                cnt = row - cnt;
            }
            ans = ans + cnt * f[col - 1 - i];
        }
        return ans;
    }


    // 866
    // 回文素数
    /*
        求出大于或等于 N 的最小回文素数。
    */
    // 优化：除了11，偶数位的回文串都能被11整除
    bool a_primePalindrome[20000];
    int p_primePalindrome[20000], total_primePalindrome;
    int f[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    int primePalindrome(int N)
    {
        if (N == 1)
        {
            return 2;
        }

        total_primePalindrome = 0;
        memset(a_primePalindrome, false, sizeof(a_primePalindrome));
        for (int i = 2; i < 20000; i++)
        {
            if (!a_primePalindrome[i])
            {
                p_primePalindrome[total_primePalindrome++] = i;
                for (int j = i * i; j < 20000; j += i)
                {
                    a_primePalindrome[j] = true;
                }
            }
        }

        if (N <= 100)
        {
            for (int i = N; ; i++)
            {
                if (do_primePalindrome1(i) && do_primePalindrome2(i))
                {
                    return i;
                }
            }
        }
        if ((N & 1) == 0)
        {
            N++;
        }
        int last = 0;
        for (int i = 0; i < 9; i++)
        {
            if (f[i] > N)
            {
                break;
            }
            last = i;
        }
        for (int i = N; ; i += 2)
        {
            if (f[last] < i)
            {
                last++;
                if (!(last & 1))
                {
                    i = f[last] - 1;
                    last++;
                    continue;
                }
            }
            if (do_primePalindrome2(i) && do_primePalindrome1(i))
            {
                return i;
            }
        }
    }
    bool do_primePalindrome1(int n)
    {
        int a[10], cnt=0;
        while (n != 0)
        {
            a[cnt++] = n % 10;
            n /= 10;
        }
        int L = 0, R = cnt -1;
        while (L < R)
        {
            if (a[L] != a[R])
            {
                return false;
            }
            L++;
            R--;
        }
        return true;
    }
    bool do_primePalindrome2(int n)
    {
        for (int i = 0; i < total_primePalindrome; i++)
        {
            if (p_primePalindrome[i] >= n)
            {
                return true;
            }
            if (n % p_primePalindrome[i] == 0)
            {
                return false;
            }
        }
        return true;
    }

    // 22
    // 括号生成
    /*
        给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
    */
    vector<string> generateParenthesis(int n)
    {
        vector<string> ans;
        do_generateParenthesis(ans, "", 0, 0, n);
        return ans;

    }
    void do_generateParenthesis(vector<string>& ans, string s, int left, int length, int n)
    {
        if (length == n * 2)
        {
            if (left == 0)
            {
                ans.push_back(s);
            }
            return;
        }
        if (left > n || left < 0)
        {
            return;
        }
        do_generateParenthesis(ans, s + "(", left + 1, length + 1, n);
        do_generateParenthesis(ans, s + ")", left - 1, length + 1, n);
    }

    // 1008
    // 先序遍历构造二叉搜索树
    /*
        返回与给定先序遍历 preorder 相匹配的二叉搜索树（binary search tree）的根结点。
    */
    TreeNode* bstFromPreorder(vector<int>& preorder)
    {
        return do_bstFromPreorder(preorder, 0, preorder.size() - 1);
    }
    TreeNode* do_bstFromPreorder(vector<int>& preorder, int L, int R)
    {
        if (L > R)
        {
            return 0;
        }
        if (L == R)
        {
            TreeNode* node = new TreeNode(preorder[L]);
            node->left = 0;
            node->right = 0;
            return node;
        }
        int index = L;
        for (int i = L + 1; i <= R; i++)
        {
            if (preorder[i] > preorder[L])
            {
                break;
            }
            index = i;
        }
        TreeNode* node = new TreeNode(preorder[L]);
        node->left = do_bstFromPreorder(preorder, L + 1, index);
        node->right = do_bstFromPreorder(preorder, index + 1, R);
        return node;
    }


    // 338
    // 比特位计数
    /*
        给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
    */
    vector<int> countBits(int num)
    {
        vector<int> ans(num + 1);
        ans[0] = 0;
        for (int i = 1; i <= num; i++)
        {
            ans[i] = ans[i & (i - 1)] + 1;
//            if ((i & 1) == 0)
//            {
//                ans[i] = ans[i >> 1];
//            }
//            else
//            {
//                ans[i] = ans[i - 1] + 1;
//            }
        }
        return ans;
    }


    // 59
    // 螺旋矩阵2
    /*
        给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
    */
    vector<vector<int> > generateMatrix(int n)
    {
        vector<vector<int> > ans(n);
        for (int i = 0; i < n; i++)
        {
            vector<int> result(n);
            for (int j = 0; j < n; j++)
            {
                result[j] = 0;
            }
            ans[i] = result;
        }
        int row = 0, col = 0, dir = 0;
        for (int i = 0; i < n * n; i++)
        {
            ans[row][col] = i + 1;
            if (dir == 0)
            {
                if (col + 1 >= n || ans[row][col + 1] != 0)
                {
                    dir = 1;
                    row++;
                }
                else
                {
                    col++;
                }
            }
            else if (dir == 1)
            {
                if (row + 1 >= n || ans[row + 1][col] != 0)
                {
                    dir = 2;
                    col--;
                }
                else
                {
                    row++;
                }
            }
            else if (dir == 2)
            {
                if (col - 1 < 0 || ans[row][col - 1] != 0)
                {
                    dir = 3;
                    row--;
                }
                else
                {
                    col--;
                }
            }
            else
            {
                if (row - 1 < 0 || ans[row - 1][col] != 0)
                {
                    dir = 0;
                    col++;
                }
                else
                {
                    row--;
                }
            }
        }
        return ans;
    }


    // 78
    // 子集
    /*
        给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
        说明：解集不能包含重复的子集。
    */
    vector<vector<int> > subsets(vector<int>& nums)
    {
        vector<vector<int> > ans;
        int len = 1 << nums.size();
        for (int i = 0; i < len; i++)
        {
            vector<int> result;
            int j = i, index = 0;
            while (j != 0)
            {
                if (j & 1)
                {
                    result.push_back(nums[index]);
                }
                j >>= 1;
                index++;
            }
            ans.push_back(result);
        }
        return ans;
    }

    // 654
    // 最大二叉树
    /*
        给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
        二叉树的根是数组中的最大元素。
        左子树是通过数组中最大值左边部分构造出的最大二叉树。
        右子树是通过数组中最大值右边部分构造出的最大二叉树。
        通过给定的数组构建最大二叉树，并且输出这个树的根节点。
    */
    TreeNode* constructMaximumBinaryTree(vector<int>& nums)
    {
        return do_constructMaximumBinaryTree(nums, 0, nums.size() - 1);
    }
    TreeNode* do_constructMaximumBinaryTree(vector<int>& nums, int L, int R)
    {
        if (L > R)
        {
            return 0;
        }
        if (L == R)
        {
            TreeNode *node = new TreeNode(nums[L]);
            node->left = 0;
            node->right = 0;
            return node;
        }
        int index = L;
        for (int i = L; i <= R; i++)
        {
            if (nums[index] < nums[i])
            {
                index = i;
            }
        }
        TreeNode *node = new TreeNode(nums[index]);
        node->left = do_constructMaximumBinaryTree(nums, L, index - 1);
        node->right = do_constructMaximumBinaryTree(nums, index + 1, R);
        return node;
    }

    // 938
    // 二叉搜索树的范围和
    /*
        给定二叉搜索树的根结点 root，返回 L 和 R（含）之间的所有结点的值的和。二叉搜索树保证具有唯一的值
    */
    bool findL_rangeSumBST, findR_rangeSumBST;
    int sum_rangeSumBST;
    int rangeSumBST(TreeNode* root, int L, int R)
    {
        if (L == R)
        {
            return L;
        }

        findL_rangeSumBST = false;
        findR_rangeSumBST = false;
        sum_rangeSumBST = 0;
        do_rangeSumBST(root, L, R);
        return sum_rangeSumBST;
    }
    void do_rangeSumBST(TreeNode *root, int L, int R)
    {
        if (findL_rangeSumBST && findR_rangeSumBST)
        {
            return;
        }
        if (root == 0)
        {
            return;
        }
        do_rangeSumBST(root->left, L, R);
        if (root->val == L)
        {
            findL_rangeSumBST = true;
        }
        if (!findR_rangeSumBST && findL_rangeSumBST)
        {
            sum_rangeSumBST += root->val;
        }
        if (root->val == R)
        {
            findR_rangeSumBST = true;
        }
        do_rangeSumBST(root->right, L, R);
    }

    // 807
    // 保持城市天际线
    /*
        在二维数组grid中，grid[i][j]代表位于某处的建筑物的高度。 我们被允许增加任何数量（不同建筑物的数量可能不同）的建筑物的高度。 高度 0 也被认为是建筑物。
        最后，从新数组的所有四个方向（即顶部，底部，左侧和右侧）观看的“天际线”必须与原始数组的天际线相同。 城市的天际线是从远处观看时，由所有建筑物形成的矩形的外部轮廓。 请看下面的例子。
        建筑物高度可以增加的最大总和是多少？
    */
    int maxIncreaseKeepingSkyline(vector<vector<int> >& grid)
    {
        int row = grid.size();
        int col = grid[0].size();
        vector<int> ma(row);
        for (int i = 0; i < row; i++)
        {
            int x = grid[i][0];
            for (int j = 0; j < col; j++)
            {
                if (grid[i][j] > x)
                {
                    x = grid[i][j];
                }
            }
            ma[i] = x;
        }
        int ans = 0;
        for (int i = 0; i < col; i++)
        {
            int x = grid[0][i];
            for (int j = 0; j < row; j++)
            {
                if (x < grid[j][i])
                {
                    x = grid[j][i];
                }
            }
            for (int j = 0; j < row; j++)
            {
                int mi = x > ma[j] ? ma[j] : x;
                ans = ans + mi - grid[j][i];
            }
        }
        return ans;
    }

    // 950
    // 按递增顺序显示卡牌
    /*
        牌组中的每张卡牌都对应有一个唯一的整数。你可以按你想要的顺序对这套卡片进行排序。
        最初，这些卡牌在牌组里是正面朝下的（即，未显示状态）。
        现在，重复执行以下步骤，直到显示所有卡牌为止：

        从牌组顶部抽一张牌，显示它，然后将其从牌组中移出。
        如果牌组中仍有牌，则将下一张处于牌组顶部的牌放在牌组的底部。
        如果仍有未显示的牌，那么返回步骤 1。否则，停止行动。
        返回能以递增顺序显示卡牌的牌组顺序。

        答案中的第一张牌被认为处于牌堆顶部。
    */
    // 提示：逆向思考，找到规律
    vector<int> deckRevealedIncreasing(vector<int>& deck)
    {
        int length = deck.size();
        vector<int> ans(length);
        sort(deck.begin(), deck.end());
        for (int i = length - 1; i >= 0; i--)
        {
            ans[i] = deck[i];
            int index = ans[length - 1];
            for (int j = length - 1; j > i + 1; j--)
            {
                ans[j] = ans[j - 1];
            }
            if (i != length - 1)
            {
                ans[i + 1] = index;
            }
        }
        return ans;
    }
};
int main()
{
    Solution* solution = new Solution();
    string s = "1*2-3/4+5*6-7*8+9/10";
    cout << solution->calculate(s);
    return 0;
}
