#include <map>
#include <set>
#include <cmath>
#include <queue>
#include <stack>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
struct TrieNode
{
    TrieNode *child[26];
    bool isWord;
};
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
    // 299
    // 猜数字游戏
    /*
        你正在和你的朋友玩 猜数字（Bulls and Cows）游戏：你写下一个数字让你的朋友猜。每次他猜测后，你给他一个提示，告诉他有多少位数字和确切位置都猜对了（称为“Bulls”, 公牛），有多少位数字猜对了但是位置不对（称为“Cows”, 奶牛）。你的朋友将会根据提示继续猜，直到猜出秘密数字。
        请写出一个根据秘密数字和朋友的猜测数返回提示的函数，用 A 表示公牛，用 B 表示奶牛。
        请注意秘密数字和朋友的猜测数都可能含有重复数字。
    */
    string getHint(string secret, string guess)
    {
        int len = secret.size();
        int v1[10], v2[10], numA = 0, numB = 0;
        memset(v1, 0, sizeof(v1));
        memset(v2, 0, sizeof(v2));
        for (int i = 0; i < len; i++)
        {
            if (secret[i] == guess[i])
            {
                numA++;
            }
            else
            {
                v1[secret[i] - '0']++;
                v2[guess[i] - '0']++;
            }
        }
        for (int i = 0; i < 10; i++)
        {
            numB += min(v1[i], v2[i]);
        }
        string A = "", B = "";
        if (numA == 0)
        {
            A = "0";
        }
        else
        {
            while (numA)
            {
                A = A + (char)(numA % 10 + '0');
                numA /= 10;
            }
            reverse(A.begin(), A.end());
        }
        if (numB == 0)
        {
            B = "0";
        }
        else
        {
            while (numB)
            {
                B = B + (char)(numB % 10 + '0');
                numB /= 10;
            }
            reverse(B.begin(), B.end());
        }
        return A + "A" + B + "B";
    }

    // 289
    // 生命游戏
    /*
        根据百度百科，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在1970年发明的细胞自动机。
        给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞具有一个初始状态 live（1）即为活细胞， 或 dead（0）即为死细胞。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：
        如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
        如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
        如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
        如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
        根据当前状态，写一个函数来计算面板上细胞的下一个（一次更新后的）状态。下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。
        进阶:
            你可以使用原地算法解决本题吗？请注意，面板上所有格子需要同时被更新：你不能先更新某些格子，然后使用它们的更新后的值再更新其他格子。
            本题中，我们使用二维数组来表示面板。原则上，面板是无限的，但当活细胞侵占了面板边界时会造成问题。你将如何解决这些问题？
    */
    void gameOfLife(vector<vector<int>> &board)
    {
        int row = board.size();
        if (row == 0)
        {
            return;
        }
        int col = board[0].size();
        if (col == 0)
        {
            return;
        }
        int dir[8][2] = {{1, -1}, {1, 0}, {1, 1}, {0, -1}, {0, 1}, {-1, -1}, {-1, 0}, {-1, 1}};
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                int cnt = 0;
                for (int k = 0; k < 8; k++)
                {
                    int x = i + dir[k][0];
                    int y = j + dir[k][1];
                    if (x >= 0 && x < row && y >= 0 && y < col && (board[x][y] == 1 || board[x][y] == 2))
                    {
                        cnt++;
                    }
                }
                if (board[i][j] == 0)
                {
                    if (cnt == 3)
                    {
                        board[i][j] = -1;
                    }
                }
                else
                {
                    if (cnt < 2 || cnt > 3)
                    {
                        board[i][j] = 2;
                    }
                }
            }
        }
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
            {
                if (board[i][j] == 2)
                {
                    board[i][j] = 0;
                }
                else if (board[i][j] == -1)
                {
                    board[i][j] = 1;
                }
            }
    }

    // 287
    // 寻找重复数
    /*
        给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
        说明：
            不能更改原数组（假设数组是只读的）。
            只能使用额外的 O(1) 的空间。
            时间复杂度小于 O(n2) 。
            数组中只有一个重复的数字，但它可能不止重复出现一次。
    */
    // 提示：排序，map，二分（小于mid的个数大于mid在左边，否则在右边.有bug），循环链表入口？
    int findDuplicate(vector<int> &nums)
    {
        int len = nums.size();
        int L = 1, R = len - 1;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            int l = 0, g = 0, e = 0;
            for (int i = 0; i < len; i++)
            {
                if (nums[i] > mid)
                {
                    g++;
                }
                else if (nums[i] < mid)
                {
                    l++;
                }
                else
                {
                    e++;
                }
            }
            if (e > 1)
            {
                return mid;
            }
            if (l >= mid)
            {
                R = mid - 1;
            }
            else
            {
                L = mid + 1;
            }
        }
        return L;
    }

    // 284
    // 顶端迭代器
    /*
        给定一个迭代器类的接口，接口包含两个方法： next() 和 hasNext()。设计并实现一个支持 peek() 操作的顶端迭代器 -- 其本质就是把原本应由 next() 方法返回的元素 peek() 出来。
        进阶：你将如何拓展你的设计？使之变得通用化，从而适应所有的类型，而不只是整数型
    */
    // 提示：抄袭的
    int PeekingIterator_IsPeek;
    int PeekingIterator_Value;
    PeekingIterator(const vector<int> &nums) : Iterator(nums)
    {
        this->PeekingIterator_IsPeek = false;
    }
    int peek()
    {
        if (this->PeekingIterator_IsPeek)
        {
            return this->PeekingIterator_Value;
        }
        this->PeekingIterator_IsPeek = true;
        return this->PeekingIterator_Value = Iterator::next();
    }
    int next()
    {
        if (this->isPeek_)
        {
            this->PeekingIterator_IsPeek = false;
            return this->PeekingIterator_Value;
        }
        return Iterator::next();
    }
    bool hashNext() const
    {
        if (this->PeekingIterator_IsPeek)
        {
            return true;
        }
        return Iterator::hasNext();
    }

    // 279
    // 完全平方数
    /*
        给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
    */
    int numSquares(int n)
    {
        int m = sqrt(n);
        vector<int> dp(n + 1, n);
        dp[0] = 0;
        for (int i = 1; i <= m; i++)
        {
            for (int j = i * i; j <= n; j++)
            {
                dp[j] = min(dp[j], dp[j - i * i] + 1);
            }
        }
        return dp[n];
    }

    // 275
    // H指数II
    /*
        给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照升序排列。编写一个方法，计算出研究者的 h 指数。
        h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）至多有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）"
        说明:
            如果 h 有多有种可能的值 ，h 指数是其中最大的那个。
        进阶：
            这是 H指数 的延伸题目，本题中的 citations 数组是保证有序的。
            你可以优化你的算法到对数时间复杂度吗？
    */
    int hIndexII(vector<int> &citations)
    {
        if (citations.size() == 0)
        {
            return 0;
        }
        int h = 1;
        for (int i = citations.size() - 1; i >= 0; i--)
        {
            if (h == citations[i])
            {
                return h;
            }
            if (h > citations[i])
            {
                return h - 1;
            }
            h++;
        }
        return h - 1;
    }

    // 274
    // H指数
    /*
        给定一位研究者论文被引用次数的数组（被引用次数是非负整数）。编写一个方法，计算出研究者的 h 指数。
        h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）至多有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）”
        说明: 如果 h 有多种可能的值，h 指数是其中最大的那个。
    */
    int hIndex(vector<int> &citations)
    {
        sort(citations.begin(), citations.end(), greater<int>()); // 倒叙排列
        int len = citations.size();
        int h = len - 1;
        for (int i = len - 1; i >= 0; i--)
        {
            if (citations[h] >= h + 1)
            {
                return h + 1;
            }
            h--;
        }
        return 0;
    }

    // 264
    // 丑数II
    /*
        编写一个程序，找出第 n 个丑数。
        丑数就是只包含质因数 2, 3, 5 的正整数。
        说明:
            1 是丑数。
            n 不超过1690。
    */
    int nthUglyNumber(int n)
    {
        vector<int> ans(n + 1);
        ans[0] = 1;
        int x = 0, y = 0, z = 0, xx, yy, zz;
        for (int i = 1; i <= n; i++)
        {
            xx = ans[x] * 2;
            yy = ans[y] * 3;
            zz = ans[z] * 5;
            int mi = min(min(xx, yy), zz);
            if (xx == mi)
            {
                x++;
            }
            if (yy == mi)
            {
                y++;
            }
            if (zz == mi)
            {
                z++;
            }
            ans[i] = mi;
        }
        return ans[n];
    }

    // 260
    // 只出现一次的数字III
    /*
        给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。
        注意：
            结果输出的顺序并不重要，对于上面的例子， [5, 3] 也是正确答案。
            你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？
    */
    vector<int> singleNumber(vector<int> &nums)
    {
        vector<int> ans(2);
        int len = nums.size(), m = 0;
        for (int i = 0; i < len; i++)
        {
            m = m ^ nums[i];
        }
        int k = 0;
        while ((m & 1) == 0)
        {
            k++;
            m >>= 1;
        }
        k = 1 << k;
        ans[0] = 0, ans[1] = 0;
        for (int i = 0; i < len; i++)
        {
            if ((nums[i] & k) == k)
            {
                ans[0] ^= nums[i];
            }
            else
            {
                ans[1] ^= nums[i];
            }
        }
        return ans;
    }

    // 241
    // 为运算表达式设计优先级
    /*
        给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。
    */
    vector<int> diffWaysToCompute(string input)
    {
        int len = input.size();
        return DFS_diffWaysToCompute(input, len, 0, len - 1);
    }
    vector<int> DFS_diffWaysToCompute(string &input, int len, int L, int R)
    {
        vector<int> ans;
        if (L > R)
        {
            return ans;
        }
        int num = 0;
        bool onlyNum = true;
        for (int i = L; i <= R; i++)
        {
            if (input[i] == '+' || input[i] == '-' || input[i] == '*')
            {
                onlyNum = false;
                vector<int> num1 = DFS_diffWaysToCompute(input, len, L, i - 1);
                vector<int> num2 = DFS_diffWaysToCompute(input, len, i + 1, R);
                for (int j = 0; j < num1.size(); j++)
                    for (int k = 0; k < num2.size(); k++)
                    {
                        switch (input[i])
                        {
                        case '+':
                            num = num1[j] + num2[k];
                            break;
                        case '-':
                            num = num1[j] - num2[k];
                            break;
                        case '*':
                            num = num1[j] * num2[k];
                            break;
                        }
                        ans.push_back(num);
                    }
            }
            else
            {
                num = num * 10 + input[i] - '0';
            }
        }
        if (onlyNum)
        {
            ans.push_back(num);
        }
        return ans;
    }

    // 240
    // 搜索二维矩阵 II
    /*
        编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
        每行的元素从左到右升序排列。
        每列的元素从上到下升序排列。
    */
    bool searchMatrixII(vector<vector<int>> &matrix, int target)
    {
        int row = matrix.size();
        if (row == 0)
        {
            return false;
        }
        int col = matrix[0].size();
        if (col == 0)
        {
            return false;
        }
        int curR = 0, curC = col - 1;
        while (curR < row && curC >= 0)
        {
            if (matrix[curR][curC] == target)
            {
                return true;
            }
            else if (matrix[curR][curC] > target)
            {
                curC--;
            }
            else if (matrix[curR][curC] < target)
            {
                curR++;
            }
        }
        return false;
    }

    // 238
    /*
        除自身以外数组的乘积
        给定长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
        说明:
            请不要使用除法，且在 O(n) 时间复杂度内完成此题。
        进阶：
            你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
    */
    // 提示：左右各扫描一次
    vector<int> productExceptSelf(vector<int> &nums)
    {
        int len = nums.size();
        vector<int> ans(len);
        int num = 1;
        for (int i = 0; i < len; i++)
        {
            num *= nums[i];
            ans[i] = num;
        }
        num = 1;
        for (int i = len - 1; i >= 0; i--)
        {
            if (i > 0)
            {
                ans[i] = ans[i - 1];
                ans[i] *= num;
            }
            else
            {
                ans[i] = num;
            }
            num *= nums[i];
        }
        return ans;
    }

    // 236
    // 二叉树的最近公共祖先
    /*
        给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
        说明:
            所有节点的值都是唯一的。
            p、q 为不同节点且均存在于给定的二叉树中。
    */
    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
    {
        if (root == 0 || root == p || root == q)
        {
            return root;
        }
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (left != 0 && right != 0)
        {
            return root;
        }
        return left != 0 ? left : right;
    }

    // 230
    // 二叉搜索树中第k小的元素
    /*
        给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
        说明：
        你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。
    */
    int kthSmallest(TreeNode *root, int k)
    {
        int cnt = 0;
        stack<TreeNode *> sta;
        while (true)
        {
            while (root != 0)
            {
                sta.push(root);
                root = root->left;
            }
            if (sta.empty())
            {
                return 0;
            }
            root = sta.top();
            sta.pop();
            cnt++;
            if (cnt == k)
            {
                return root->val;
            }
            root = root->right;
        }
    }

    // 229
    // 求众数
    /*
        给定一个大小为 n 的数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。
        说明: 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1)。
    */
    vector<int> majorityElement(vector<int> &nums)
    {
        int len = nums.size();
        vector<int> ans;
        int num1 = 0, num2 = 0;
        int cnt1 = 0, cnt2 = 0;
        for (int i = 0; i < len; i++)
        {
            if ((num1 == nums[i] || cnt1 == 0) && nums[i] != num2)
            {
                num1 = nums[i];
                cnt1++;
            }
            else if (num2 == nums[i] || cnt2 == 0)
            {
                num2 = nums[i];
                cnt2++;
            }
            else
            {
                cnt1--;
                cnt2--;
            }
        }
        cnt1 = 0;
        cnt2 = 0;
        for (int i = 0; i < len; i++)
        {
            cnt1 += (nums[i] == num1) ? 1 : 0;
            cnt2 += (nums[i] == num2) ? 1 : 0;
        }
        if (cnt1 > len / 3)
        {
            ans.push_back(num1);
        }
        if (cnt2 > len / 3 && (num1 != num2))
        {
            ans.push_back(num2);
        }
        return ans;
    }

    // 228
    // 汇总区间
    /*
        给定一个无重复元素的有序整数数组，返回数组区间范围的汇总。
        输入: [0,1,2,4,5,7]
        输出: ["0->2","4->5","7"]
    */
    // 注意：有负数，注意越界
    vector<string> summaryRanges(vector<int> &nums)
    {
        int len = nums.size();
        vector<string> ans;
        for (int i = 0; i < len; i++)
        {
            int j = i + 1;
            while (j < len && (long long)nums[j] - nums[j - 1] == 1)
            {
                j++;
            }
            j--;
            if (i != j)
            {
                ans.push_back(NumToString_summaryRanges(nums[i]) + "->" + NumToString_summaryRanges(nums[j]));
            }
            else
            {
                ans.push_back(NumToString_summaryRanges(nums[i]));
            }
            i = j;
        }
        return ans;
    }
    string NumToString_summaryRanges(long long num)
    {
        if (num == 0)
        {
            return "0";
        }
        int dir = 1;
        if (num < 0)
        {
            num = -num;
            dir = -1;
        }
        string ans = "";
        while (num != 0)
        {
            ans = ans + (char)(num % 10 + '0');
            num /= 10;
        }
        int L = 0, R = ans.size() - 1;
        while (L < R)
        {
            swap(ans[L], ans[R]);
            L++;
            R--;
        }
        if (dir == -1)
        {
            return "-" + ans;
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
                    num.push(Cal_calculate(num1, num2, op));
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
                    num.push(Cal_calculate(num1, num2, op));
                }
                ops.push(s[i]);
            }
            else if (s[i] >= '0' && s[i] <= '9')
            {
                long long k = 0, j = i;
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
            num.push(Cal_calculate(num1, num2, op));
        }
        return num.top();
    }
    int Cal_calculate(int num1, int num2, char op)
    {
        switch (op)
        {
        case '+':
            return num1 + num2;
        case '-':
            return num1 - num2;
        case '*':
            return num1 * num2;
        case '/':
            return num1 / num2;
        }
        return 0;
    }

    // 223
    // 矩形面积
    /*
        在二维平面上计算出两个由直线构成的矩形重叠后形成的总面积。
        每个矩形由其左下顶点和右上顶点坐标表示，如图所示。
        说明: 假设矩形面积不会超出 int 的范围。
    */
    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H)
    {
        int leftX = max(A, E);
        int leftY = min(D, H);
        int rightX = min(C, G);
        int rightY = max(B, F);
        long long ans = (long long)(C - A) * (D - B) + (G - E) * (H - F);
        if (leftX <= rightX && leftY >= rightY)
        {
            ans = ans - (rightX - leftX) * (leftY - rightY);
        }
        return ans;
    }

    // 222
    // 完全二叉树的节点个数
    /*
        给出一个完全二叉树，求出该树的节点个数。
    */
    int countNodes(TreeNode *root)
    {
        if (root == 0)
        {
            return 0;
        }
        return 1 + countNodes(root->left) + countNodes(root->right);
    }

    // 221
    // 最大正方形
    /*
        在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。
    */
    // 提示: 为1的时候dp[i][j] = 1 + min(dp[i - 1][j], min(dp[i - 1][j - 1], dp[i][j - 1]));为0的时候dp[i][j]为0
    int maximalSquare(vector<vector<char>> &matrix)
    {
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
        vector<vector<int>> dp(row, vector<int>(col));
        int ans = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (matrix[i][j] == '1')
                {
                    if (i == 0 || j == 0)
                    {
                        dp[i][j] = 1;
                    }
                    else
                    {
                        dp[i][j] = 1 + min(dp[i - 1][j], min(dp[i - 1][j - 1], dp[i][j - 1]));
                    }
                    ans = max(ans, dp[i][j]);
                }
            }
        }
        return ans * ans;
    }

    // 220
    // 存在重复元素 III
    /*
        给定一个整数数组，判断数组中是否有两个不同的索引 i 和 j，使得 nums [i] 和 nums [j] 的差的绝对值最大为 t，并且 i 和 j 之间的差的绝对值最大为 ķ。
    */
    // 提示：可以维护一个长度为k的有序的数组，每次减一个加一个还是有序的。时间复杂度就会为nlogk
    // 当遍历到nums[i]时，找到该数组中第一个大于等于nums[i]-t的数。如果满足要求返回true。
    bool containsNearbyAlmostDuplicate(vector<int> &nums, int k, int t)
    {
        int len = nums.size();
        set<long> Set;
        for (int i = 0; i < len; i++)
        {
            auto it = Set.lower_bound(nums[i] - (long)t);
            if (it != Set.end() && *it - (long)nums[i] <= t)
            {
                return true;
            }
            Set.insert(nums[i]);
            if (Set.size() > k)
            {
                Set.erase(nums[i - k]);
            }
        }
        return false;
    }

    // 216
    // 组合总和III
    /*
        找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
        说明：
        所有数字都是正整数。
        解集不能包含重复的组合。
    */
    vector<vector<int>> combinationSum3(int k, int n)
    {
        vector<vector<int>> ans;
        vector<int> result(k);
        DFS_combinationSum3(ans, n, k, result, 0, 0, 1);
        return ans;
    }
    void DFS_combinationSum3(vector<vector<int>> &ans, int n, int k, vector<int> &result, int sum, int stp, int sta)
    {
        if (sum > n || stp > k)
        {
            return;
        }
        if (stp == k)
        {
            if (sum == n)
            {
                ans.push_back(result);
            }
            return;
        }
        for (int i = sta; i < 10; i++)
        {
            if (sum + i > n)
            {
                break;
            }
            result[stp] = i;
            DFS_combinationSum3(ans, n, k, result, sum + i, stp + 1, i + 1);
        }
    }

    // 215
    // 数组中的第k个最大元素
    /*
        在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
    */
    // 方法1： 快排
    // 方法2： 维护topk（优先队列）
    int findKthLargest(vector<int> &nums, int k)
    {
        int ans = INT_MIN;
        QuickSort_findKthLargest(nums, nums.size() - k, 0, nums.size() - 1, ans);
        return ans;
    }
    void QuickSort_findKthLargest(vector<int> &nums, int k, int L, int R, int &ans)
    {
        if (ans != INT_MIN || L > R)
        {
            return;
        }
        int LL = L, RR = R;
        int key = nums[L];
        while (L < R)
        {
            while (L < R && nums[R] >= key)
            {
                R--;
            }
            if (L == R)
            {
                break;
            }
            nums[L++] = nums[R];
            while (L < R && nums[L] <= key)
            {
                L++;
            }
            if (L == R)
            {
                break;
            }
            nums[R--] = nums[L];
        }
        nums[L] = key;
        if (L == k)
        {
            ans = nums[L];
            return;
        }
        QuickSort_findKthLargest(nums, k, LL, L - 1, ans);
        QuickSort_findKthLargest(nums, k, L + 1, RR, ans);
        return;
    }

    // 213
    // 打家劫舍 II
    /*
        你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
        给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
    */
    int rob(vector<int> &nums)
    {
        int len = nums.size();
        if (len == 0)
        {
            return 0;
        }
        if (len == 1)
        {
            return nums[0];
        }
        vector<vector<int>> dp(len, vector<int>(2));
        int ans = 0;
        dp[1][0] = 0;
        dp[1][1] = nums[1];
        for (int i = 2; i < len; i++)
        {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }
        ans = max(dp[len - 1][0], dp[len - 1][1]);
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        for (int i = 1; i < len - 1; i++)
        {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }
        ans = max(ans, max(dp[len - 2][0], dp[len - 2][1]));
        return ans;
    }

    // 211
    // 添加与搜索单词 - 数据结构设计
    /*
        设计一个支持以下两种操作的数据结构：
            void addWord(word)
            bool search(word)
        search(word) 可以搜索文字或正则表达式字符串，字符串只包含字母 . 或 a-z 。 . 可以表示任何一个字母。
    */
    // 提示：用hash保存长度一样的字符串集
    map<int, vector<string>> WordDictionary_Map;
    WordDictionary()
    {
    }
    void addWord(string word)
    {
        int len = word.size();
        WordDictionary_Map[len].push_back(word);
    }
    bool search(string word)
    {
        int len = word.size();
        vector<string> words = WordDictionary_Map[len];
        int total = words.size();
        for (int i = 0; i < total; i++)
        {
            bool eq = true;
            for (int j = 0; j < len; j++)
            {
                if (!(word[j] == words[i][j] || word[j] == '.'))
                {
                    eq = false;
                    break;
                }
            }
            if (eq)
            {
                return true;
            }
        }
        return false;
    }

    // 210
    // 课程表II
    /*
        现在你总共有 n 门课需要选，记为 0 到 n-1。
        在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
        给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
        可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。
    */
    vector<int> findOrder(int numCourses, vector<vector<int>> &prerequisites)
    {
        vector<int> ans(numCourses);
        vector<int> num(numCourses, 0);
        queue<int> Q;
        int index = 0;
        for (int i = 0; i < prerequisites.size(); i++)
        {
            num[prerequisites[i][0]]++;
        }

        int finished = 0;
        for (int i = 0; i < numCourses; i++)
        {
            if (num[i] == 0)
            {
                Q.push(i);
                ans[index++] = i;
                finished++;
            }
        }
        while (!Q.empty())
        {
            int f = Q.front();
            Q.pop();
            for (int i = 0; i < prerequisites.size(); i++)
            {
                if (num[prerequisites[i][0]] == 0)
                {
                    continue;
                }
                if (prerequisites[i][1] == f)
                {
                    num[prerequisites[i][0]]--;
                    if (num[prerequisites[i][0]] == 0)
                    {
                        finished++;
                        Q.push(prerequisites[i][0]);
                        ans[index++] = prerequisites[i][0];
                    }
                }
            }
            if (finished == numCourses)
            {
                return ans;
            }
        }
        if (finished == numCourses)
        {
            return ans;
        }
        ans.resize(0);
        return ans;
    }

    // 209
    // 长度最小的子数组
    /*
        给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组。如果不存在符合条件的连续子数组，返回 0。
    */
    int minSubArrayLen(int s, vector<int> &nums)
    {
        int len = nums.size();
        int mi = len + 1;
        int last = 0, sum = 0;
        for (int i = 0; i < len; i++)
        {
            sum = sum + nums[i];
            while (sum >= s)
            {
                mi = min(mi, i - last + 1);
                sum -= nums[last];
                last++;
            }
        }
        return mi == len + 1 ? 0 : mi;
    }

    // 208
    // 实现 Trie (前缀树)
    /*
        实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。
    */
    // 注意：app，apple这种单词，所以需要isWord
    /*
    TrieNode()
    {
        for (int i = 0; i < 26; i++)
        {
            child[i] = nullptr;
        }
        isWord = false;
    }
    void insert(string word)
    {
        TrieNode *node = this;
        int index;
        for (int i = 0; i < word.size(); i++)
        {
            index = word[i] - 'a';
            if (node->child[index] == 0)
            {
                node->child[index] = new TrieNode();
            }
            node = node->child[index];
        }
        node->isWord = true;
    }
    bool search(string word)
    {
        TrieNode *node = this;
        int index;
        for (int i = 0; i < word.size(); i++)
        {
            index = word[i] - 'a';
            if (node->child[index] == 0)
            {
                return false;
            }
            node = node->child[index];
        }
        return node->isWord;
    }
    bool startsWith(string prefix)
    {
        TrieNode *node = this;
        int index;
        for (int i = 0; i < prefix.size(); i++)
        {
            index = prefix[i] - 'a';
            if (node->child[index] == 0)
            {
                return false;
            }
            node = node->child[index];
        }
        return true;
    }
    */

    // 207
    // 课程表
    /*
        现在你总共有 n 门课需要选，记为 0 到 n-1。
        在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
        给定课程总量以及它们的先决条件，判断是否可能完成所有课程的学习？
    */
    // 方法2： DFS检查是否有环，有环则不行
    bool canFinish(int numCourses, vector<vector<int>> &prerequisites)
    {
        vector<int> num(numCourses, 0);
        queue<int> Q;
        for (int i = 0; i < prerequisites.size(); i++)
        {
            num[prerequisites[i][0]]++;
        }

        int finished = 0;
        for (int i = 0; i < numCourses; i++)
        {
            if (num[i] == 0)
            {
                Q.push(i);
                finished++;
            }
        }
        while (!Q.empty())
        {
            int f = Q.front();
            Q.pop();
            for (int i = 0; i < prerequisites.size(); i++)
            {
                if (num[prerequisites[i][0]] == 0)
                {
                    continue;
                }
                if (prerequisites[i][1] == f)
                {
                    num[prerequisites[i][0]]--;
                    if (num[prerequisites[i][0]] == 0)
                    {
                        finished++;
                        Q.push(prerequisites[i][0]);
                    }
                }
            }
            if (finished == numCourses)
            {
                return true;
            }
        }
        return finished == numCourses;
    }

    // 201
    // 数字范围按位与
    /*
        给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。
    */

    // 200
    // 岛屿数量
    /*
        给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。
    */
    int numIslands(vector<vector<char>> &grid)
    {
        int ans = 0;
        int row = grid.size();
        if (row == 0)
        {
            return ans;
        }
        int col = grid[0].size();
        if (col == 0)
        {
            return ans;
        }
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (grid[i][j] == '1')
                {
                    ans++;
                    DFS_numIslands(grid, row, col, i, j);
                }
            }
        }
        return ans;
    }
    void DFS_numIslands(vector<vector<char>> &grid, int row, int col, int x, int y)
    {
        if (x >= 0 && x < row && y >= 0 && y < col && grid[x][y] == '1')
        {
            int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            grid[x][y] = '2';
            for (int i = 0; i < 4; i++)
            {
                int xx = x + dir[i][0];
                int yy = y + dir[i][1];
                DFS_numIslands(grid, row, col, xx, yy);
            }
        }
    }
};
int main()
{
    Solution *solution = new Solution();
    return 0;
}
