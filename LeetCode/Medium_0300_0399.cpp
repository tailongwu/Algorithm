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
    // 397
    // 整数替换
    /*
        给定一个正整数 n，你可以做如下操作：
        1. 如果 n 是偶数，则用 n / 2替换 n。
        2. 如果 n 是奇数，则可以用 n + 1或n - 1替换 n。
        n 变为 1 所需的最小替换次数是多少？
    */
    // 提示：主要是应该+1还是-1.如果（n-1）能被4整数就n-1否则n+1。3单独判断
    int integerReplacement(int n)
    {
        long long m = n;
        int ans = 0;
        while (m != 1)
        {
            if (m == 3)
            {
                ans += 2;
                break;
            }
            if (!(m & 1))
            {
                m >>= 1;
            }
            else if ((m & 3) == 1)
            {
                m--;
            }
            else
            {
                m++;
            }
            ans++;
        }
        return ans;
    }


    // 396
    // 旋转函数
    /*
        给定一个长度为 n 的整数数组 A 。
        假设 Bk 是数组 A 顺时针旋转 k 个位置后的数组，我们定义 A 的“旋转函数” F 为：
        F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]。
        计算F(0), F(1), ..., F(n-1)中的最大值。
        注意:
        可以认为 n 的值小于 100000
    */
    // 提示：Fk-F0可以公式求出
    int maxRotateFunction(vector<int>& A)
    {
        int len = A.size();
        long sum1 = 0, sum2 = 0, ans = 0, last = 0;
        for (int i = 0; i < len; i++)
        {
            sum1 += A[i];
            sum2 += i * A[i];
        }
        ans = sum2;
        for (int i = 1; i < len; i++)
        {
            last += A[len - i];
            ans = max(ans, sum1 * i - len * last + sum2);
        }
        return ans;
    }


    // 395
    // 至少有k个重复字符的最长子串
    /*
        找到给定字符串（由小写字符组成）中的最长子串 T ， 要求 T 中的每一字符出现次数都不少于 k 。输出 T 的长度。
    */
    // 提示：分段计数有问题，比如"bbaaacbd"，3。分段后，继续用该算法。
    int longestSubstring(string s, int k)
    {
        int len = s.size();
        int num[26] = {0};
        for (int i = 0; i < len; i++)
        {
            num[s[i] - 'a']++;
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            if (num[s[i] - 'a'] >= k)
            {
                int j = i;
                string str = "";
                while (j < len && num[s[j] - 'a'] >= k)
                {
                    str += s[j];
                    j++;
                }
                if (i == 0 && j == len)
                {
                    ans = max(ans, j - i);
                }
                else
                {
                    ans = max (ans, longestSubstring(str, k));
                }
                i = j;
            }
        }
        return ans;
    }


    // 394
    // 字符串解码
    /*
        给定一个经过编码的字符串，返回它解码后的字符串。
        编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
        你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
        此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
        s = "3[a]2[bc]", 返回 "aaabcbc";s = "3[a2[c]]", 返回 "accaccacc"
    */
    string decodeString(string s)
    {
        int len = s.size();
        return DFS_decodeString(s, 0);
    }
    string DFS_decodeString(string s, int sta)
    {
        if (sta == s.size())
        {
            return "";
        }
        if (s[sta] >= '0' && s[sta] <= '9')
        {
            int num = 0;
            while (sta < s.size() && s[sta] >= '0' && s[sta] <= '9')
            {
                num = num * 10 + s[sta] - '0';
                sta++;
            }
            string ans = "";
            string str = "[";
            int cnt = 1;
            sta++;
            while (sta < s.size() && cnt != 0)
            {
                if (s[sta] == '[')
                {
                    cnt++;
                }
                else if (s[sta] == ']')
                {
                    cnt--;
                }
                str += s[sta];
                sta++;
            }
            string subs = DFS_decodeString(str, 0);
            for (int i = 0; i < num; i++)
            {
                ans += subs;
            }
            str = "";
            for (int i = sta; i < s.size(); i++)
            {
                str += s[i];
            }
            return ans + DFS_decodeString(str, 0);
        }
        if (s[sta] == '[' || s[sta] == ']')
        {
            sta++;
            return DFS_decodeString(s, sta);
        }
        string str = "";
        while (sta < s.size() && s[sta] != '[' && s[sta] != ']' && !(s[sta] >= '0' && s[sta] <= '9'))
        {
            str += s[sta];
            sta++;
        }
        return str + DFS_decodeString(s, sta);
    }


    // 392
    // 判断子序列
    /*
        给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
        你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。
        字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
    */
    bool isSubsequence(string s, string t)
    {
        int len1 = s.size(), len2 = t.size(), index1 = 0, index2 = 0;
        while (index1 < len1 && index2 < len2)
        {
            if (s[index1] == t[index2])
            {
                index1++;
            }
            index2++;
        }
        return index1 == len1;
    }


    // 390
    // 消除游戏
    /*
        给定一个从1 到 n 排序的整数列表。
        首先，从左到右，从第一个数字开始，每隔一个数字进行删除，直到列表的末尾。
        第二步，在剩下的数字中，从右到左，从倒数第一个数字开始，每隔一个数字进行删除，直到列表开头。
        我们不断重复这两步，从左到右和从右到左交替进行，直到只剩下一个数字。
        返回长度为 n 的列表中，最后剩下的数字
    */
    // 提示：f[n]=2*(n/2+1-f[n/2]);
    int lastRemaining(int n)
    {
        return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
    }


    // 386
    // 字典序排数
    /*
        给定一个整数 n, 返回从 1 到 n 的字典顺序。
        例如，
        给定 n =1 3，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9] 。
        请尽可能的优化算法的时间复杂度和空间复杂度。 输入的数据 n 小于等于 5,000,000
    */
    vector<int> lexicalOrder(int n)
    {
        vector<int> ans;
        DFS_lexicalOrder(ans, n, 0);
        return ans;
    }
    void DFS_lexicalOrder(vector<int> &ans, int n, int cur)
    {
        for (int i = 0; i < 10; i++)
        {
            if (cur * 10 + i > n)
            {
                break;
            }
            if (cur * 10 + i == 0)
            {
                continue;
            }
            ans.push_back(cur * 10 + i);
            DFS_lexicalOrder(ans, n, cur * 10 + i);
        }
    }


    // 378
    // 有序矩阵中第K小的元素
    /*
        给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第k小的元素。
        请注意，它是排序后的第k小元素，而不是第k个元素。
    */
    // 提示：时间复杂度N*logN*logX，注意有重复元素
    int kthSmallest(vector<vector<int> >& matrix, int k)
    {
        int row = matrix.size();
        if (row == 0)
        {
            return 0;
        }
        int col = matrix.size();
        if (col == 0)
        {
            return 0;
        }
        int L = matrix[0][0], R = matrix[row - 1][col - 1];
        while (L < R)
        {
            int mid = (L + R) >> 1, cnt = 0;
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    if (matrix[i][j] <= mid)
                    {
                        cnt++;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (cnt < k)
            {
                L = mid + 1;
            }
            else
            {
                R = mid;
            }
        }
        return L;
    }


    // 377
    // 组合总和IV
    /*
        给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。
        进阶：
            如果给定的数组中含有负数会怎么样？
            问题会产生什么变化？
            我们需要在题目中添加什么限制来允许负数的出现？
    */
    // 提示：dfs会超时
    int combinationSum4(vector<int>& nums, int target)
    {
        int len = nums.size();
        vector<unsigned int> dp(target + 1, 0);
        dp[0] = 1;
        for (int i = 0; i <= target; i++)
        {
            if (dp[i] == 0)
            {
                continue;
            }
            for (int j = 0; j < len; j++)
            {
                if (i + nums[j] <= target)
                {
                    dp[i + nums[j]] += dp[i];
                }
            }
        }
        return dp[target];
    }


    // 376
    // 摆动序列
    /*
        如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。
        给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。
        进阶：你能否用 O(n) 时间复杂度完成此题?
    */
    int wiggleMaxLength(vector<int>& nums)
    {
        int len = nums.size();
        vector<int> num;
        if (len == 0)
        {
            return 0;
        }
        num.push_back(nums[0]);
        for (int i = 1; i < len; i++)
        {
            if (nums[i] != nums[i - 1])
            {
                num.push_back(nums[i]);
            }
        }
        len = num.size();
        if (len <= 2)
        {
            return len;
        }
        vector<int> dp(len);
        dp[0] = 1;
        dp[1] = 2;
        int ans = 2;
        for (int i = 2; i < len; i++)
        {
            if ((num[i] - num[i - 1]) * (num[i - 1] - num[i - 2]) < 0)
            {
                dp[i] = dp[i - 1] + 1;
            }
            else
            {
                dp[i] = dp[i - 1];
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }


    // 375
    // 猜数字大小II
    /*
        我们正在玩一个猜数游戏，游戏规则如下：
        我从 1 到 n 之间选择一个数字，你来猜我选了哪个数字。
        每次你猜错了，我都会告诉你，我选的数字比你的大了或者小了。
        然而，当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。直到你猜到我选的数字，你才算赢得了这个游戏。
    */
    // 提示：dp[i][j] = min(dp[i][j], max(dp[i][k + 1], dp[k - 1][j]) + k)
    int getMoneyAmount(int n)
    {
        if (n <= 2)
        {
            return n - 1;
        }
        vector<vector<int> > dp(n + 1, vector<int>(n + 1));
        for (int i = 1; i <= n; i++)
        {
            for (int j = i; j > 0; j--)
            {
                dp[i][j] = n * n;
                if (i == j)
                {
                    dp[i][j] = 0;
                }
                else if (i == j + 1)
                {
                    dp[i][j] = j;
                }
                else
                {
                    for (int k = j + 1; k < i; k++)
                    {
                        dp[i][j] = min(dp[i][j], max(dp[i][k + 1], dp[k - 1][j]) + k);
                    }
                }
            }
        }
        return dp[n][1];
    }


    // 373
    // 查找和最小的k对数字
    /*
        给定两个以升序排列的整形数组 nums1 和 nums2, 以及一个整数 k。
        定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2。
        找到和最小的 k 对数字 (u1,v1), (u2,v2) ... (uk,vk)。
    */
    struct P_kSmallestPairs
    {
        int a, b, sum;
    };
    static bool cmp_kSmallestPairs(const P_kSmallestPairs &x, const P_kSmallestPairs &y)
    {
        return x.sum < y.sum;
    }
    vector<vector<int> > kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k)
    {
        int len1 = nums1.size(), len2 = nums2.size();
        k = min(k, len1 * len2);
        vector<vector<int> > ans(k);
        vector<P_kSmallestPairs> p;
        P_kSmallestPairs node;
        for (int i = 0; i < len1; i++)
        {
            for (int j = 0; j < len2; j++)
            {
                node.a = nums1[i];
                node.b = nums2[j];
                node.sum = nums1[i] + nums2[j];
                p.push_back(node);
            }
        }
        sort(p.begin(), p.end(), cmp_kSmallestPairs);
        vector<int> result(2);
        for (int i = 0; i < k; i++)
        {
            result[0] = p[i].a;
            result[1] = p[i].b;
            ans[i] = result;
        }
        return ans;
    }


    // 372
    // 超级次方
    /*
        你的任务是计算 ab 对 1337 取模，a 是一个正整数，b 是一个非常大的正整数且会以数组形式给出。
    */
    int superPow(int a, vector<int>& b)
    {
        int len = b.size(), last = a % 1337, ans = 1;
        for (int i = len - 1; i >= 0; i--)
        {
            ans = ans * Do_superPow(last, b[i]) % 1337;
            last = Do_superPow(last, 10);
        }
        return ans;
    }
    int Do_superPow(int a, int b)
    {
        int ans = 1;
        while (b)
        {
            if (b & 1)
            {
                ans = ans * a % 1337;
            }
            a = a * a % 1337;
            b >>= 1;
        }
        return ans;
    }


    // 368
    // 最大整除子集
    /*
        给出一个由无重复的正整数组成的集合，找出其中最大的整除子集，子集中任意一对 (Si，Sj) 都要满足：Si % Sj = 0 或 Sj % Si = 0。
        如果有多个目标子集，返回其中任何一个均可。
    */
    vector<int> largestDivisibleSubset(vector<int>& nums)
    {
        sort(nums.begin(), nums.end());
        int len = nums.size(), index = 0, mx = 1;
        vector<int> f(len, -1);
        vector<int> dp(len, 1);
        vector<int> ans;
        if (len == 0)
        {
            return ans;
        }
        for (int i = len - 1; i >= 0; i--)
        {
            for (int j = i; j < len; j++)
            {
                if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1)
                {
                    dp[i] = dp[j] + 1;
                    f[i] = j;
                    if (dp[i] > mx)
                    {
                        mx = dp[i];
                        index = i;
                    }
                }
            }
        }
        while (index != -1)
        {
            ans.push_back(nums[index]);
            if (index == f[index])
            {
                break;
            }
            index = f[index];
        }
        return ans;
    }


    // 365
    // 水壶问题
    /*
        有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
        如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
        你允许：
            装满任意一个水壶
            清空任意一个水壶
            从一个水壶向另外一个水壶倒水，直到装满或者倒空
    */
    bool canMeasureWater(int x, int y, int z)
    {
        if (x + y < z)
        {
            return false;
        }
        while (y != 0)
        {
            int r = x % y;
            x = y;
            y = r;
        }
        if (x == 0)
        {
            return z == 0;
        }
        return z % x == 0;
    }


    // 357
    // 计算各个位数不同的数字个数
    /*
        给定一个非负整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10n 。
    */
    int countNumbersWithUniqueDigits(int n)
    {
        if (n >= 10)
        {
            n = 10;
        }
        int ans = 0;
        for (int i = 1; i <= n; i++)
        {
            int k = 1;
            for (int j = 9; j > 9 - (i - 1); j--)
            {
                k *= j;
            }
            ans = ans + 9 * k;
        }
        return ans;
    }


    // 347
    // 前 K 个高频元素
    /*
        给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
        说明：
            你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
            你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。
    */
    vector<int> topKFrequent(vector<int>& nums, int k)
    {

    }


    // 343
    // 整数拆分
    /*
        给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
        你可以假设 n 不小于 2 且不大于 58。
    */
    int integerBreak(int n)
    {
        if (n <= 3)
        {
            return n - 1;
        }
        int dp[60];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for (int i = 4; i <= n; i++)
        {
            dp[i] = max(dp[i - 2] * 2, dp[i - 3] * 3);
        }
        return dp[n];
    }


    // 338
    // 比特位计数
    /*
        给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
        进阶:
            给出时间复杂度为O(n*sizeof(integer))的解答非常容易。但你可以在线性时间O(n)内用一趟扫描做到吗？
            要求算法的空间复杂度为O(n)。
            你能进一步完善解法吗？要求在C++或任何其他语言中不使用任何内置函数（如 C++ 中的 __builtin_popcount）来执行此操作。
    */
    vector<int> countBits(int num)
    {
        vector<int> ans(num + 1);
        ans[0] = 0;
        for (int i = 1; i <= num; i++)
        {
            ans[i] = ans[i >> 1] + (i & 1);
        }
        return ans;
    }


    // 337
    // 打家劫舍III
    /*
        在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
        计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
    */
    int rob(TreeNode* root)
    {
        return DFS_rob(root, false);
    }
    int DFS_rob(TreeNode *node, bool robbed)
    {
        if (node == 0)
        {
            return 0;
        }
        if (robbed)
        {
            return DFS_rob(node->left, false) + DFS_rob(node->right, false);
        }
        else
        {
            int result1 = DFS_rob(node->left, true) + DFS_rob(node->right, true) + node->val;
            int result2 = DFS_rob(node->left, false) + DFS_rob(node->right, false);
            return result1 > result2 ? result1 : result2;
        }
    }


    // 334
    // 递增的三元子序列
    /*
        给定一个未排序的数组，判断这个数组中是否存在长度为 3 的递增子序列。
        数学表达式如下:
        如果存在这样的 i, j, k,  且满足 0 ≤ i < j < k ≤ n-1，
        使得 arr[i] < arr[j] < arr[k] ，返回 true ; 否则返回 false 。
        说明:
            要求算法的时间复杂度为 O(n)，空间复杂度为 O(1) 。
    */
    // 提示：保存最小的和次小的，b被赋值过代表前面有比它小的数
    bool increasingTriplet(vector<int>& nums)
    {
        int len = nums.size();
        if (len < 3)
        {
            return false;
        }
        int a = nums[0], b = INT_MAX;
        for (int i = 1; i < len; i++)
        {
            if (nums[i] <= a)
            {
                a = nums[i];
            }
            else
            {
                if (nums[i] > b)
                {
                    return true;
                }
                else
                {
                    b = nums[i];
                }
            }
        }
        return false;
    }


    // 331
    // 验证二叉树的前序序列化
    /*
        序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录。
        给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。
        每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。
        你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 "1,,3" 。
    */
    bool isValidSerialization(string preorder)
    {
        int len = preorder.size(), cnt = 1;
        if (len == 0)
        {
            return true;
        }
        for (int i = 0; i < len; i++)
        {
            if (preorder[i] == '#')
            {
                cnt--;
            }
            else if (preorder[i] >= '0' && preorder[i] <= '9')
            {
                while (i < len && preorder[i] >= '0' && preorder[i] <= '9')
                {
                    i++;
                }
                i--;
                cnt++;
            }
            else
            {
                continue;
            }
            if (cnt < 1)
            {
                if (i != len - 1)
                {
                    return false;
                }
                return cnt == 0;
            }
        }
        return cnt == 0;
    }


    // 328
    // 奇偶链表
    /*
        给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。
        请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。
        说明:
            应当保持奇数节点和偶数节点的相对顺序。
            链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。
    */
    ListNode* oddEvenList(ListNode* head)
    {
        if (head == 0)
        {
            return head;
        }
        ListNode *head1 = head, *head2 = head->next;
        ListNode *node1 = head1, *node2 = head2, *node = head, *newHead, *temp;
        int index = 0;
        while (node != 0)
        {
            temp = node;
            node = node->next;
            if (!(index & 1))
            {
                node1->next = temp;
                node1 = temp;
            }
            else
            {
                node2->next = temp;
                node2 = temp;
            }
            index++;
        }
        node1->next = 0;
        if (node2 != 0)
        {
            node2->next = 0;
        }
        newHead = head1;
        node1->next = head2;
        return newHead;
    }


    // 324
    // 摆动排序II
    /*
        给定一个无序的数组 nums，将它重新排列成 nums[0] < nums[1] > nums[2] < nums[3]... 的顺序。
        说明:
            你可以假设所有输入都会得到有效的结果。
        进阶:
            你能用 O(n) 时间复杂度和 / 或原地 O(1) 额外空间来实现吗？
    */
    // 提示：排序再穿插；用快排思想找到中位数，然后小的放在0，2，4...，大的放在1，3，5等；
    void wiggleSort(vector<int>& nums)
    {

    }


    // 322
    // 零钱兑换
    /*
        给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
    */
    int coinChange(vector<int>& coins, int amount)
    {
        int len = coins.size();
        sort(coins.begin(), coins.end());
        vector<int> dp(amount + 1, -1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++)
        {
            for (int j = 0; j < len; j++)
            {
                if (i < coins[j])
                {
                    break;
                }
                if (dp[i - coins[j]] != -1 && (dp[i] == -1 || dp[i] > dp[i - coins[j]] + 1))
                {
                    dp[i] = dp[i - coins[j]] + 1;
                }
            }
        }
        return dp[amount];
    }


    // 319
    // 灯泡开关
    /*
        初始时有 n 个灯泡关闭。 第 1 轮，你打开所有的灯泡。 第 2 轮，每两个灯泡你关闭一次。 第 3 轮，每三个灯泡切换一次开关（如果关闭则开启，如果开启则关闭）。第 i 轮，每 i 个灯泡切换一次开关。 对于第 n 轮，你只切换最后一个灯泡的开关。 找出 n 轮后有多少个亮着的灯泡。
    */
    int bulbSwitch(int n)
    {
        int L = 1, R = n, ans = 0;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            if ((long long)mid * mid > n)
            {
                R = mid - 1;
            }
            else
            {
                ans = mid;
                L = mid + 1;
            }
        }
        return ans;
    }


    // 318
    // 最大单词长度乘积
    /*
        给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。
    */
    int maxProduct(vector<string>& words)
    {
        int len = words.size();
        vector<int> ha(len, 0);
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < words[i].size(); j++)
            {
                ha[i] = ha[i] | (1 << (words[i][j] - 'a'));
            }
        }
        int ma = 0;
        for (int i = 0; i < len; i++)
        {
            for (int j = i + 1; j < len; j++)
            {
                if ((ha[i] & ha[j]) == 0)
                {
                    int k = words[i].size() * words[j].size();
                    ma = max(k, ma);
                }
            }
        }
        return ma;
    }


    // 313
    // 超级丑数
    /*
        编写一段程序来查找第 n 个超级丑数。
        超级丑数是指其所有质因数都是长度为 k 的质数列表 primes 中的正整数。
    */
    int nthSuperUglyNumber(int n, vector<int>& primes)
    {
        int len = primes.size();
        vector<int> num(len, 0);
        vector<int> ans(n);
        ans[0] = 1;
        for (int i = 1; i < n; i++)
        {
            int mi = ans[num[0]] * primes[0];
            for (int j = 1; j < len; j++)
            {
                mi = min(mi, ans[num[j]] * primes[j]);
            }
            for (int j = 0; j < len; j++)
            {
                if (mi == ans[num[j]] * primes[j])
                {
                    num[j]++;
                }
            }
            ans[i] = mi;
        }
        return ans[n - 1];
    }


    // 310
    // 最小高度树
    /*
        对于一个具有树特征的无向图，我们可选择任何一个节点作为根。图因此可以成为树，在所有可能的树中，具有最小高度的树被称为最小高度树。给出这样的一个图，写出一个函数找到所有的最小高度树并返回他们的根节点。
        格式
        该图包含 n 个节点，标记为 0 到 n - 1。给定数字 n 和一个无向边 edges 列表（每一个边都是一对标签）。
        你可以假设没有重复的边会出现在 edges 中。由于所有的边都是无向边， [0, 1]和 [1, 0] 是相同的，因此不会同时出现在 edges 里。
        说明:
            根据树的定义，树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。
            树的高度是指根节点和叶子节点之间最长向下路径上边的数量。
    */
    // 提示：每次删除叶子节点
    vector<int> findMinHeightTrees(int n, vector<vector<int> >& edges)
    {

    }


    // 309
    // 最佳买卖股票时机含冷冻期
    /*
        给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
        设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
        你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    */
    int maxProfit(vector<int>& prices)
    {
        int len = prices.size();
        if (len == 0)
        {return 0;}
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


    // 307
    // 区域和检索 - 数组可修改
    /*
        给定一个整数数组  nums，求出数组从索引 i 到 j  (i ≤ j) 范围内元素的总和，包含 i,  j 两点。
        update(i, val) 函数可以通过将下标为 i 的数值更新为 val，从而对数列进行修改。
    */



    // 306
    // 累加数
    /*
        累加数是一个字符串，组成它的数字可以形成累加序列。
        一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
        给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。
        说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况
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


    // 304
    // 二维区域和检索 - 矩阵不可变
    /*
        给定一个二维矩阵，计算其子矩形范围内元素的总和，该子矩阵的左上角为 (row1, col1) ，右下角为 (row2, col2)。
        说明:
            你可以假设矩阵不可变。
            会多次调用 sumRegion 方法。
            你可以假设 row1 ≤ row2 且 col1 ≤ col2。
    */
    vector<vector<int> > sumNumMatrix;
    NumMatrix(vector<vector<int> >& matrix)
    {
        int row = matrix.size(), col = 0;
        if (row > 0)
        {
            col = matrix[0].size();
        }
        sumNumMatrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            sumNumMatrix[i].resize(col);
            int sum = 0;
            for (int j = 0; j < col; j++)
            {
                sum = sum + matrix[i][j];
                sumNumMatrix[i][j] = sum + (i > 0 ? sumNumMatrix[i - 1][j] : 0);
            }
        }
    }
    int sumRegion(int row1, int col1, int row2, int col2)
    {
        int sum = sumNumMatrix[row2][col2];
        if (row1 > 0)
        {
            sum -= sumNumMatrix[row1 - 1][col2];
        }
        if (col1 > 0)
        {
            sum -= sumNumMatrix[row2][col1 - 1];
        }
        if (row1 > 0 && col1 > 0)
        {
            sum += sumNumMatrix[row1 - 1][col1 - 1];
        }
        return sum;
    }


    // 300
    // 最长上升子序列
    /*
        给定一个无序的整数数组，找到其中最长上升子序列的长度。
        说明:
            可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
            你算法的时间复杂度应该为 O(n2) 。
        进阶:
            你能将算法的时间复杂度降低到 O(n log n) 吗?
    */
    // 提示：ans不是最长的序列，但是长度是最长的
    int lengthOfLIS(vector<int>& nums)
    {
        int len = nums.size();
        vector<int> ans;
        for (int i = 0; i < len; i++)
        {
            int si = ans.size();
            if (si == 0 || nums[i] > ans[si - 1])
            {
                ans.push_back(nums[i]);
            }
            else
            {
                int L = 0, R = si - 1, index = 0;
                while (L <= R)
                {
                    int mid = (L + R) >> 1;
                    if (ans[mid] >= nums[i])
                    {
                        index = mid;
                        R = mid - 1;
                    }
                    else
                    {
                        L = mid + 1;
                    }
                }
                ans[index] = nums[i];
            }
        }
        return ans.size();
    }
};
int main()
{
    Solution* solution = new Solution();
    return 0;
}
