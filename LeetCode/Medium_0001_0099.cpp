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
    // 98
    // 验证二叉搜索树
    /*
        给定一个二叉树，判断其是否是一个有效的二叉搜索树。
        假设一个二叉搜索树具有如下特征：
        节点的左子树只包含小于当前节点的数。
        节点的右子树只包含大于当前节点的数。
        所有左子树和右子树自身必须也是二叉搜索树。
    */
    // 提示：中序遍历是增序的
    bool isValidBST(TreeNode *root)
    {
        long long last = LONG_MIN;
        return DFS_isValidBST(root, last);
    }
    bool DFS_isValidBST(TreeNode *node, long long &last)
    {
        if (node == 0)
        {
            return true;
        }
        bool left = DFS_isValidBST(node->left, last);
        if (node->val <= last || !left)
        {
            return false;
        }
        last = node->val;
        bool right = DFS_isValidBST(node->right, last);
        return left && right;
    }

    // 96
    // 不同的二叉搜索树
    /*
        给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
    */
    // 提示：以1为根节点，左边0个右边n-1个；以2为根节点，左边1个右边n-2个；。。。以n为根节点，左边n-1个，右边0个。
    // ans[n] = ans[0]*ans[n-1]+ans[1]*ans[n-2]+ans[2]*ans[n-3]+...+ans[n-2]*ans[1]+ans[n-1]*ans[0]
    // 该数是卡特兰数，h(n)=C(2n,n)/(n+1)或者h(n)=c(2n,n)-c(2n,n-1)
    int numTrees(int n)
    {
        vector<int> dp(n + 1);
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++)
        {
            dp[i] = 0;
            for (int j = 0; j < i; j++)
            {
                dp[i] = dp[i] + dp[j] * dp[i - 1 - j];
            }
        }
        return dp[n];
    }

    // 95
    // 不同的二叉搜索树II
    /*
        给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。
    */
    vector<TreeNode *> generateTrees(int n)
    {
        if (n == 0)
        {
            vector<TreeNode *> ans;
            return ans;
        }
        return DFS_generateTrees(1, n);
    }
    vector<TreeNode *> DFS_generateTrees(int sta, int en)
    {
        vector<TreeNode *> result;
        if (sta > en)
        {
            result.push_back(0);
            return result;
        }
        for (int i = sta; i <= en; i++)
        {
            vector<TreeNode *> left = DFS_generateTrees(sta, i - 1);
            vector<TreeNode *> right = DFS_generateTrees(i + 1, en);
            for (int j = 0; j < left.size(); j++)
            {
                for (int k = 0; k < right.size(); k++)
                {
                    TreeNode *root = new TreeNode(i);
                    root->left = left[j];
                    root->right = right[k];
                    result.push_back(root);
                }
            }
        }
        return result;
    }

    // 94
    // 二叉树的中序遍历
    /*
        给定一个二叉树，返回它的中序 遍历。
    */
    vector<int> inorderTraversal(TreeNode *root)
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

    // 93
    // 复原IP地址
    /*
        给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
    */
    // 提示：4个for循环；dfs；
    vector<string> restoreIpAddresses(string s)
    {
        vector<string> ans;
        int len = s.size();
        DFS_restoreIpAddresses(ans, s, "", len, 0, 0);
        return ans;
    }
    void DFS_restoreIpAddresses(vector<string> &ans, string s, string result, int len, int sta, int stp)
    {
        if (stp == 3)
        {
            int ip = 0;
            string ip_s = "";
            for (int i = sta; i < len; i++)
            {
                ip = ip * 10 + s[i] - '0';
                if (ip > 255)
                {
                    break;
                }
                if (ip == 0 && len - 1 > sta)
                {
                    ip = 256;
                    break;
                }
                ip_s += s[i];
            }
            if (ip < 256)
            {
                ans.push_back(result + "." + ip_s);
            }
            return;
        }
        for (int i = sta; i < len - 1; i++)
        {
            int ip = 0;
            string ip_s = "";
            if (sta != 0)
            {
                ip_s = ".";
            }
            for (int j = sta; j <= i; j++)
            {
                ip = ip * 10 + s[j] - '0';
                if (ip == 0 && i > j)
                {
                    ip = 256;
                    break;
                }
                ip_s += s[j];
                if (ip > 255)
                {
                    break;
                }
            }
            if (ip > 255)
            {
                break;
            }
            DFS_restoreIpAddresses(ans, s, result + ip_s, len, i + 1, stp + 1);
        }
    }

    // 92
    // 反转链表II
    /*
        反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
        说明:
        1 ≤ m ≤ n ≤ 链表长度。
    */
    ListNode *reverseBetween(ListNode *head, int m, int n)
    {
        ListNode *last = 0, *node = head;
        for (int i = 0; i < m - 1; i++)
        {
            last = node;
            node = node->next;
        }
        ListNode *newHead, *cur, *right = node;
        for (int i = m; i <= n; i++)
        {
            cur = node->next;
            node->next = newHead;
            newHead = node;
            node = cur;
        }
        if (right != 0)
        {
            right->next = node;
        }
        if (last != 0)
        {
            last->next = newHead;
        }
        else
        {
            head = newHead;
        }
        return head;
    }

    // 91
    // 解码方法
    /*
        一条包含字母 A-Z 的消息通过以下方式进行了编码：
        'A' -> 1
        'B' -> 2
        ...
        'Z' -> 26
        给定一个只包含数字的非空字符串，请计算解码方法的总数。
    */
    // 提示：dp或者dfs，dp效率更高，可以只用3个变量
    int numDecodings(string s)
    {
        int len = s.size();
        int num;
        vector<int> dp(len + 1);
        dp[0] = 1;
        for (int j = 0; j < len; j++)
        {
            int i = j + 1;
            if (i >= 2)
            {
                num = (s[j - 1] - '0') * 10 + s[j] - '0';
                if (num < 27 && num > 9)
                {
                    dp[i] += dp[i - 2];
                }
            }
            num = s[j] - '0';
            if (num > 0)
            {
                dp[i] += dp[i - 1];
            }
        }
        return dp[len];
        //        int ans = 0;
        //        DFS_numDecodings(s, ans, s.size(), 0);
        //        return ans;
    }
    //    void DFS_numDecodings(string &s, int &ans, int len, int sta)
    //    {
    //        if (sta == len)
    //        {
    //            ans++;
    //            return;
    //        }
    //        int num = 0;
    //        for (int i = sta; i < len; i++)
    //        {
    //            num = num * 10 + s[i] - '0';
    //            if (num > 26 || num == 0)
    //            {
    //                break;
    //            }
    //            if (num > 0)
    //            {
    //                DFS_numDecodings(s, ans, len, i + 1);
    //            }
    //        }
    //    }

    // 90
    // 子集II
    /*
        给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
        说明：解集不能包含重复的子集。
    */
    vector<vector<int>> subsetsWithDup(vector<int> &nums)
    {
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        vector<int> result(nums.size());
        DFS_subsetsWithDup(nums, ans, result, nums.size(), 0, 0);
        return ans;
    }
    void DFS_subsetsWithDup(vector<int> &nums, vector<vector<int>> &ans, vector<int> &result, int len, int sta, int stp)
    {
        vector<int> a(stp);
        for (int i = 0; i < stp; i++)
        {
            a[i] = result[i];
        }
        ans.push_back(a);
        for (int i = sta; i < len; i++)
        {
            if (i > sta && nums[i] == nums[i - 1])
            {
                continue;
            }
            result[stp] = nums[i];
            DFS_subsetsWithDup(nums, ans, result, len, i + 1, stp + 1);
        }
    }

    // 89
    // 格雷编码
    /*
        格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
        给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。格雷编码序列必须以 0 开头。
    */
    // 提示：格雷编码生成，证明方法？
    vector<int> grayCode(int n)
    {
        int len = 1 << n;
        vector<int> ans(len);
        for (int i = 0; i < len; i++)
        {
            ans[i] = i ^ (i >> 1);
        }
        return ans;
        //        int len = 1 << n;
        //        vector<int> ans(len);
        //        map<int, bool> m;
        //        string last = "";
        //        int lastNum = 0;
        //        for (int i = 0; i < n; i++)
        //        {
        //            last += '0';
        //        }
        //        m[lastNum] = true;
        //        ans[0] = lastNum;
        //        for (int i = 1; i < len; i++)
        //        {
        //            for (int j = 0; j < n; j++)
        //            {
        //                int newNum;
        //                if (last[j] == '0')
        //                {
        //                    newNum = lastNum + (1 << (n - 1 - j));
        //                    if (!m[newNum])
        //                    {
        //                        last[j] = '1';
        //                        ans[i] = newNum;
        //                        lastNum = newNum;
        //                        m[newNum] = true;
        //                        break;
        //                    }
        //                }
        //                else
        //                {
        //                    newNum = lastNum - (1 << (n - 1 - j));
        //                    if (!m[newNum])
        //                    {
        //                        last[j] = '0';
        //                        ans[i] = newNum;
        //                        lastNum = newNum;
        //                        m[newNum] = true;
        //                        break;
        //                    }
        //                }
        //            }
        //        }
        //        return ans;
    }

    // 86
    // 分隔链表
    /*
        给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
        你应当保留两个分区中每个节点的初始相对位置。
    */
    ListNode *partition(ListNode *head, int x)
    {
        ListNode *newHead = 0, *leftHead = 0, *rightHead = 0, *left = 0, *right = 0, *node = head;
        while (node != 0)
        {
            if (node->val < x)
            {
                if (leftHead == 0)
                {
                    leftHead = node;
                    left = node;
                }
                else
                {
                    left->next = node;
                    left = node;
                }
            }
            else
            {
                if (rightHead == 0)
                {
                    rightHead = node;
                    right = node;
                }
                else
                {
                    right->next = node;
                    right = node;
                }
            }
            node = node->next;
        }
        if (leftHead != 0)
        {
            newHead = leftHead;
            left->next = rightHead;
        }
        else
        {
            newHead = rightHead;
        }
        if (right != 0)
        {
            right->next = 0;
        }
        return newHead;
    }

    // 82
    // 删除排序链表中的重复元素 II
    /*
        给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
    */
    ListNode *deleteDuplicates(ListNode *head)
    {
        ListNode *newHead = 0, *newNode = 0, *node = head;
        while (node != 0)
        {
            if (node->next != 0 && node->next->val == node->val)
            {
                while (node->next != 0 && node->val == node->next->val)
                {
                    node = node->next;
                }
                node = node->next;
            }
            else
            {
                if (newHead == 0)
                {
                    newHead = node;
                    newNode = node;
                }
                else
                {
                    newNode->next = node;
                    newNode = node;
                }
                node = node->next;
            }
        }
        if (newNode != 0)
        {
            newNode->next = 0;
        }
        return newHead;
    }

    // 81
    // 搜索旋转排序数组 II
    /*
        假设按照升序排序的数组在预先未知的某个点上进行了旋转。
        ( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
        编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。
    */
    bool searchII(vector<int> &nums, int target)
    {
        int L = 0, R = nums.size() - 1;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            if (nums[mid] == target)
            {
                return true;
            }
            if (nums[L] == nums[mid])
            {
                L++;
            }
            else if (nums[L] < nums[mid])
            {
                if (target >= nums[L] && target < nums[mid])
                {
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }
            else
            {
                if (target > nums[mid] && target <= nums[R])
                {
                    L = mid + 1;
                }
                else
                {
                    R = mid - 1;
                }
            }
        }
        return false;
    }

    // 80
    // 删除排序数组中的重复项II
    /*
        给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
        不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
    */
    int removeDuplicates(vector<int> &nums)
    {
        int len = nums.size();
        int index = 0, cur = 0;
        while (cur < len)
        {
            int k = cur;
            while (cur + 1 < len && nums[cur] == nums[cur + 1])
            {
                cur++;
            }
            if (cur - k >= 1)
            {
                nums[index++] = nums[k];
                nums[index++] = nums[k];
            }
            else
            {
                nums[index++] = nums[k];
            }
            cur++;
        }
        return index;
    }

    // 79
    // 单词搜索
    /*
        给定一个二维网格和一个单词，找出该单词是否存在于网格中。
        单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
    */
    bool exist(vector<vector<char>> &board, string word)
    {
        if (word.size() == 0)
        {
            return true;
        }
        if (board.size() == 0)
        {
            return false;
        }
        if (board[0].size() == 0)
        {
            return false;
        }
        int row = board.size();
        int col = board[0].size();
        int len = word.size();
        bool exist = false;
        for (int i = 0; i < row && !exist; i++)
        {
            for (int j = 0; j < col && !exist; j++)
            {
                if (board[i][j] == word[0])
                {
                    char c = board[i][j];
                    board[i][j] = 0;
                    DFS_exist(board, word, row, col, len, i, j, 1, exist);
                    board[i][j] = c;
                }
            }
        }
        return exist;
    }
    void DFS_exist(vector<vector<char>> &board, string word, int row, int col, int len, int x, int y, int stp, bool &exist)
    {
        if (exist)
        {
            return;
        }
        if (stp == len)
        {
            exist = true;
            return;
        }
        int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int xx, yy;
        for (int i = 0; i < 4; i++)
        {
            xx = x + dir[i][0];
            yy = y + dir[i][1];
            if (xx < row && xx >= 0 && yy < col && yy >= 0 && board[xx][yy] == word[stp])
            {
                char c = board[xx][yy];
                board[xx][yy] = 0;
                DFS_exist(board, word, row, col, len, xx, yy, stp + 1, exist);
                board[xx][yy] = c;
            }
        }
    }

    // 78
    // 子集
    /*
        给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
        说明：解集不能包含重复的子集。
    */
    vector<vector<int>> subsets(vector<int> &nums)
    {
        vector<vector<int>> ans;
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

    // 77
    // 组合
    /*
        给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
    */
    vector<vector<int>> combine(int n, int k)
    {
        vector<vector<int>> ans;
        vector<int> result(k);
        DFS_combine(ans, result, n, k, 1, 0);
        return ans;
    }
    void DFS_combine(vector<vector<int>> &ans, vector<int> &result, int n, int k, int sta, int step)
    {
        if (step == k)
        {
            ans.push_back(result);
            return;
        }
        if (k - step > n - sta + 1)
        {
            return;
        }
        for (int i = sta; i <= n; i++)
        {
            result[step] = i;
            DFS_combine(ans, result, n, k, i + 1, step + 1);
        }
    }

    // 75
    // 颜色分类
    /*
        给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
        此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
        注意:
        不能使用代码库中的排序函数来解决这道题。
        进阶：
        一个直观的解决方案是使用计数排序的两趟扫描算法。
        首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
        你能想出一个仅使用常数空间的一趟扫描算法吗？
    */
    // 提示：三指针法
    void sortColors(vector<int> &nums)
    {
        int p0 = 0, p1 = 0, p2 = nums.size() - 1;
        while (p0 <= p2 && p1 <= p2)
        {
            while (p0 <= p2 && nums[p0] == 0)
            {
                p0++;
                p1++;
            }
            while (p0 <= p2 && nums[p2] == 2)
            {
                p2--;
            }
            if (p0 <= p2 && nums[p1] == 0)
            {
                swap(nums[p0], nums[p1]);
                p0++;
                p1++;
            }
            else if (p0 <= p2 && nums[p1] == 2)
            {
                swap(nums[p1], nums[p2]);
                p2--;
            }
            else
            {
                p1++;
            }
        }
    }

    // 搜索二维矩阵
    // 74
    /*
        编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
        每行中的整数从左到右按升序排列。
        每行的第一个整数大于前一行的最后一个整数。
    */
    // 提示：从右上角开始
    bool searchMatrix(vector<vector<int>> &matrix, int target)
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
        int curRow = 0, curCol = col - 1;
        while (curRow < row && curCol >= 0)
        {
            if (matrix[curRow][curCol] == target)
            {
                return true;
            }
            if (matrix[curRow][curCol] < target)
            {
                curRow++;
            }
            else
            {
                curCol--;
            }
        }
        return false;
    }

    // 73
    // 矩阵置零
    /*
        给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法，使用常数空间。
    */
    // 提示：找到第一个为0的，那么其它的0分解成第一个0所在的行和列记录。最后再遍历。
    void setZeroes(vector<vector<int>> &matrix)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        int zeroRow = -1, zeroCol = -1;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (matrix[i][j] == 0)
                {
                    if (zeroRow == -1)
                    {
                        zeroRow = i;
                        zeroCol = j;
                    }
                    else
                    {
                        matrix[i][zeroCol] = 0;
                        matrix[zeroRow][j] = 0;
                    }
                }
            }
        }
        if (zeroRow != -1)
        {
            for (int i = 0; i < row; i++)
            {
                if (matrix[i][zeroCol] == 0 && i != zeroRow)
                {
                    for (int j = 0; j < col; j++)
                    {
                        matrix[i][j] = 0;
                    }
                }
            }
            for (int i = 0; i < col; i++)
            {
                if (matrix[zeroRow][i] == 0 && i != zeroCol)
                {
                    for (int j = 0; j < row; j++)
                    {
                        matrix[j][i] = 0;
                    }
                }
            }
            for (int i = 0; i < row; i++)
            {
                matrix[i][zeroCol] = 0;
            }
            for (int i = 0; i < col; i++)
            {
                matrix[zeroRow][i] = 0;
            }
        }
    }

    // 71
    // 简化路径
    /*
        以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。
        在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：Linux / Unix中的绝对路径 vs 相对路径
        请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。
    */
    // 提示：注意点和其它字符组合或者超过两个点算文件名
    string simplifyPath(string path)
    {
        int len = path.size();
        stack<string> folders;
        for (int i = 0; i < len; i++)
        {
            if (path[i] == '/')
            {
                continue;
            }
            else
            {
                int j = i;
                string folder = "";
                while (j < len && path[j] != '/')
                {
                    folder += path[j];
                    j++;
                }
                i = j - 1;
                if (folder == ".")
                {
                    continue;
                }
                else if (folder == "..")
                {
                    if (!folders.empty())
                    {
                        folders.pop();
                    }
                }
                else
                {
                    folders.push(folder);
                }
            }
        }
        int total = folders.size();
        vector<string> folder(total);
        while (!folders.empty())
        {
            folder[--total] = folders.top();
            folders.pop();
        }
        if (folder.size() == 0)
        {
            return "/";
        }
        string ans = "";
        for (int i = 0; i < folder.size(); i++)
        {
            ans = ans + "/" + folder[i];
        }
        return ans;
    }

    // 64
    // 最小路径和
    /*
        给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
        说明：每次只能向下或者向右移动一步。
    */
    int minPathSum(vector<vector<int>> &grid)
    {
        int row = grid.size();
        int col = grid[0].size();
        if (row == 0 || col == 0)
        {
            return 0;
        }
        vector<vector<int>> dp(row, vector<int>(col));
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                dp[i][j] = grid[i][j];
                if (i > 0 && j > 0)
                {
                    dp[i][j] += min(dp[i - 1][j], dp[i][j - 1]);
                }
                else if (i > 0)
                {
                    dp[i][j] += dp[i - 1][j];
                }
                else if (j > 0)
                {
                    dp[i][j] += dp[i][j - 1];
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    // 63
    // 不同路径II
    /*
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
        现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
    */
    int uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid)
    {
        int row = obstacleGrid.size();
        int col = obstacleGrid[0].size();
        if (row == 0 || col == 0 || obstacleGrid[0][0] == 1)
        {
            return 0;
        }
        vector<vector<long long>> dp(row, vector<long long>(col));
        dp[0][0] = 1;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (i + j != 0)
                {
                    dp[i][j] = 0;
                }
                if (obstacleGrid[i][j] == 0)
                {
                    if (i > 0)
                    {
                        dp[i][j] += dp[i - 1][j];
                    }
                    if (j > 0)
                    {
                        dp[i][j] += dp[i][j - 1];
                    }
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    // 62
    // 不同路径
    /*
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
        问总共有多少条不同的路径？
        说明：m 和 n 的值均不超过 100。
    */
    // 提示：ans=C(m+n-2,n-1)，实际上100和50会越界，但是没有该样例。也可以分解因式求或者杨辉三角。
    int uniquePaths(int m, int n)
    {
        int a = m + n - 2;
        int b = n - 1;
        if (a - b < b)
        {
            b = a - b;
        }
        long long ans = 1;
        int i = a, j = b;
        while (i > a - b || j > 0)
        {
            if (i > a - b)
            {
                ans *= i;
                i--;
            }
            if (j > 0)
            {
                if (ans % j == 0)
                {
                    ans /= j;
                    j--;
                }
            }
        }
        return ans;
    }

    // 61
    // 旋转链表
    /*
        给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
    */
    ListNode *rotateRight(ListNode *head, int k)
    {
        if (head == 0 || k == 0)
        {
            return head;
        }

        int len = 0, cnt = 0;
        ListNode *node = head, *last = head, *newHead;
        while (node != 0)
        {
            len++;
            last = node;
            node = node->next;
        }
        k = k % len;
        if (k == 0)
        {
            return head;
        }

        last->next = head;
        node = head;
        while (true)
        {
            cnt++;
            if (cnt == len - k)
            {
                newHead = node->next;
                node->next = 0;
                return newHead;
            }
            node = node->next;
        }
    }

    // 60
    // 第k个排列
    /*
        给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。给定 n 和 k，返回第 k 个排列。
    */
    string getPermutation(int n, int k)
    {
        vector<bool> vis(10, false);
        int f[10];
        f[0] = 1;
        for (int i = 1; i <= n; i++)
        {
            f[i] = f[i - 1] * i;
        }
        string ans = "";
        for (int i = 0; i < n; i++)
        {
            int d = 0, cnt = 0;
            while (k > f[n - 1 - i])
            {
                d++;
                k -= f[n - 1 - i];
            }
            for (int j = 1; j <= n; j++)
            {
                if (!vis[j])
                {
                    if (cnt == d)
                    {
                        vis[j] = true;
                        ans = ans + (char)(j + '0');
                    }
                    cnt++;
                }
            }
        }
        return ans;
    }

    // 59
    // 螺旋矩阵II
    /*
        给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
    */
    vector<vector<int>> generateMatrix(int n)
    {
        vector<vector<int>> ans(n, vector<int>(n));
        int index = 0, total = n * n;
        for (int i = 0; i < n && index != total; i++)
        {
            for (int j = i; j < n - i && index != total; j++)
            {
                index++;
                ans[i][j] = index;
            }
            for (int j = i + 1; j < n - i && index != total; j++)
            {
                index++;
                ans[j][n - i - 1] = index;
            }
            for (int j = n - i - 2; j >= i; j--)
            {
                index++;
                ans[n - i - 1][j] = index;
            }
            for (int j = n - i - 2; j > i && index != total; j--)
            {
                index++;
                ans[j][i] = index;
            }
        }
        return ans;
    }

    // 56
    // 合并区间
    /*
        给出一个区间的集合，请合并所有重叠的区间。
    */
    vector<vector<int>> merge(vector<vector<int>> &intervals)
    {
        int len = intervals.size();
        vector<vector<int>> ans;
        if (len == 0)
        {
            return ans;
        }
        // sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) { return a[0] < b[0]; });
        int sta = intervals[0][0], en = intervals[0][1];
        for (int i = 1; i < len; i++)
        {
            if (intervals[i][0] <= en)
            {
                if (intervals[i][1] >= en)
                {
                    en = intervals[i][1];
                }
            }
            else
            {
                vector<int> result;
                result.push_back(sta);
                result.push_back(en);
                ans.push_back(result);
                sta = intervals[i][0];
                en = intervals[i][1];
            }
        }
        vector<int> result;
        result.push_back(sta);
        result.push_back(en);
        ans.push_back(result);
        return ans;
    }

    // 55
    // 跳跃游戏
    /*
        给定一个非负整数数组，你最初位于数组的第一个位置。
        数组中的每个元素代表你在该位置可以跳跃的最大长度。
        判断你是否能够到达最后一个位置。
    */
    bool canJump(vector<int> &nums)
    {
        int len = nums.size(), maxIndex = 0;
        for (int i = 0; i < len; i++)
        {
            if (i > maxIndex)
            {
                break;
            }
            maxIndex = max(maxIndex, i + nums[i]);
        }
        return maxIndex >= len - 1;

        // 方法2
        //        int step = 1;
        //        for (int i = nums.size() - 2; i >= 0; i--)
        //        {
        //            if (nums[i] >= step)
        //            {
        //                step = 1;
        //            }
        //            else
        //            {
        //                step++;
        //            }
        //        }
        //        return step == 1;
    }

    // 54
    // 螺旋矩阵
    /*
        给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
    */
    // 提示：注意转角处包含在上次遍历
    vector<int> spiralOrder(vector<vector<int>> &matrix)
    {
        vector<int> ans;
        int row = matrix.size();
        if (row == 0)
        {
            return ans;
        }

        int col = matrix[0].size();
        int total = row * col;
        ans.resize(total);
        int index = 0;
        for (int i = 0; i < row && index != total; i++)
        {
            for (int j = i; j < col - i && index != total; j++)
            {
                ans[index++] = matrix[i][j];
            }

            for (int j = i + 1; j < row - i && index != total; j++)
            {
                ans[index++] = matrix[j][col - 1 - i];
            }

            for (int j = col - i - 2; j >= i && index != total; j--)
            {
                ans[index++] = matrix[row - 1 - i][j];
            }

            for (int j = row - i - 2; j > i && index != total; j--)
            {
                ans[index++] = matrix[j][i];
            }
        }
        return ans;
    }

    // 50
    // Pow(x, n)
    /*
        实现 pow(x, n) ，即计算 x 的 n 次幂函数。
    */
    double myPow(double x, int n)
    {
        long long nn = n;
        if (nn < 0)
        {
            nn = -nn;
        }
        double ans = 1.0;
        while (nn)
        {
            if (nn & 1)
            {
                ans = ans * x;
            }
            x = x * x;
            nn >>= 1;
        }
        if (n < 0)
        {
            return 1.0 / ans;
        }
        else
        {
            return ans;
        }
    }

    // 49
    // 字母异位词分组
    /*
        给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
    */
    vector<vector<string>> groupAnagrams(vector<string> &strs)
    {
        int len = strs.size();
        vector<vector<string>> ans;
        vector<long long> pri(26);
        int index = 0;
        for (int i = 2; i < 120; i++)
        {
            bool div = false;
            for (int j = 2; j < i; j++)
            {
                if (i % j == 0)
                {
                    div = true;
                    break;
                }
            }
            if (!div)
            {
                pri[index++] = i;
                if (index == 26)
                {
                    break;
                }
            }
        }
        map<unsigned long long, int> m;
        for (int i = 0; i < len; i++)
        {
            unsigned long long ha = 1;
            for (int j = 0; j < strs[i].size(); j++)
            {
                ha = ha * pri[strs[i][j] - 'a'];
            }
            if (m[ha] == 0)
            {
                vector<string> result;
                result.push_back(strs[i]);
                ans.push_back(result);
                m[ha] = ans.size();
            }
            else
            {
                ans[m[ha] - 1].push_back(strs[i]);
            }
        }
        return ans;
    }

    // 48
    // 旋转图像
    /*
        给定一个 n × n 的二维矩阵表示一个图像。
        将图像顺时针旋转 90 度。
        说明：
        你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
    */
    // 注意：最右边那个数不用旋转（即第二个循环的-1）
    // 方法2：先延副对角线对称，然后水平线对称。
    // 原坐标为(x,y)。旋转后坐标为(y,n-x)；对称后为(n-y,n-x)，水平对称后为(y,n-x)。
    void rotate(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        for (int i = 0; i < (n >> 1); i++)
        {
            for (int j = i; j < n - i - 1; j++)
            {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = tmp;
            }
        }
    }

    // 47
    // 全排列II
    /*
        给定一个可包含重复数字的序列，返回所有不重复的全排列。
    */
    // 提示：进行到第i位，如果大于i位且相等并使用过的，那么第i位就是重复的
    vector<vector<int>> permuteUnique(vector<int> &nums)
    {
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        vector<int> result(nums.size());
        vector<bool> vis(nums.size());
        DFS_permuteUnique(ans, nums, vis, result, 0);
        return ans;
    }
    void DFS_permuteUnique(vector<vector<int>> &ans, vector<int> &nums, vector<bool> &vis, vector<int> &result, int step)
    {
        if (step == nums.size())
        {
            ans.push_back(result);
            return;
        }
        for (int i = 0; i < nums.size(); i++)
        {
            if (!vis[i])
            {
                bool exist = false;
                for (int j = i + 1; j < nums.size(); j++)
                {
                    if (vis[j] && nums[j] == nums[i])
                    {
                        exist = true;
                        break;
                    }
                }
                if (exist)
                {
                    continue;
                }
                vis[i] = true;
                result[step] = nums[i];
                DFS_permuteUnique(ans, nums, vis, result, step + 1);
                vis[i] = false;
            }
        }
    }

    // 46
    // 全排列
    /*
        给定一个没有重复数字的序列，返回其所有可能的全排列。
    */
    bool v_permute[20];
    int result_permute[20];
    vector<vector<int>> permute(vector<int> &nums)
    {
        memset(v_permute, false, sizeof(v_permute));
        vector<vector<int>> ans;
        DFS_permute(nums, nums.size(), 0, ans);
        return ans;
    }
    void DFS_permute(vector<int> &nums, int length, int step, vector<vector<int>> &ans)
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
                DFS_permute(nums, length, step + 1, ans);
                v_permute[i] = false;
            }
        }
    }

    // 43
    // 字符串相乘
    /*
        给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
        说明：
            num1 和 num2 的长度小于110。
            num1 和 num2 只包含数字 0-9。
            num1 和 num2 均不以零开头，除非是数字 0 本身。
            不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。
    */
    // 注意："123" * "0"
    string multiply(string num1, string num2)
    {
        int len1 = num1.size(), len2 = num2.size(), index = 0, ma = 0;
        vector<char> ans(len1 + len2, 0);
        for (int i = len1 - 1; i >= 0; i--)
        {
            int r = 0, sta = index;
            for (int j = len2 - 1; j >= 0; j--)
            {
                int u = r + (num2[j] - '0') * (num1[i] - '0') + ans[sta];
                int v = u % 10;
                r = u / 10;
                ans[sta++] = u % 10;
            }
            if (r != 0)
            {
                ans[sta++] = r;
            }
            if (sta > ma)
            {
                ma = sta;
            }
            index++;
        }
        string result = "";
        for (int i = ma - 1; i >= 0; i--)
        {
            if (ans[i] == 0)
            {
                continue;
            }
            else
            {
                while (i >= 0)
                {
                    result = result + (char)(ans[i--] + '0');
                }
            }
        }
        if (result.size() == 0)
        {
            result = "0";
        }
        return result;
    }

    // 40
    // 组合总和II
    /*
        给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
        candidates 中的每个数字在每个组合中只能使用一次。
    */
    vector<vector<int>> combinationSum2(vector<int> &candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        vector<int> selections(target + 1);
        DFS_combinationSum2(ans, candidates, selections, target, 0, 0, 0);
        return ans;
    }
    void DFS_combinationSum2(vector<vector<int>> &ans, vector<int> &candidates, vector<int> &selections, int target, int sum, int step, int sta)
    {
        if (target < sum)
        {
            return;
        }
        if (target == sum)
        {
            vector<int> result(step);
            for (int i = 0; i < step; i++)
            {
                result[i] = selections[i];
            }
            bool eq = false;
            for (int i = 0; i < ans.size(); i++)
            {
                if (result.size() == ans[i].size())
                {
                    int j = 0;
                    for (; j < result.size(); j++)
                    {
                        if (result[j] != ans[i][j])
                        {
                            break;
                        }
                    }
                    if (j == result.size())
                    {
                        eq = true;
                        break;
                    }
                }
            }
            if (!eq)
            {
                ans.push_back(result);
            }
            return;
        }
        for (int i = sta; i < candidates.size(); i++)
        {
            if (sum + candidates[i] > target)
            {
                break;
            }
            selections[step] = candidates[i];
            DFS_combinationSum2(ans, candidates, selections, target, sum + candidates[i], step + 1, i + 1);
        }
    }

    // 39
    // 组合总和
    /*
        给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
        candidates 中的数字可以无限制重复被选取。
        说明：
        所有数字（包括 target）都是正整数。
        解集不能包含重复的组合。
    */
    vector<vector<int>> combinationSum(vector<int> &candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        vector<int> selections(target + 1);
        DFS_combinationSum(ans, candidates, selections, target, 0, 0, 0);
        return ans;
    }
    void DFS_combinationSum(vector<vector<int>> &ans, vector<int> &candidates, vector<int> &selections, int target, int sum, int step, int sta)
    {
        if (target < sum)
        {
            return;
        }
        if (target == sum)
        {
            vector<int> result(step);
            for (int i = 0; i < step; i++)
            {
                result[i] = selections[i];
            }
            ans.push_back(result);
            return;
        }
        for (int i = sta; i < candidates.size(); i++)
        {
            if (sum + candidates[i] > target)
            {
                break;
            }
            selections[step] = candidates[i];
            DFS_combinationSum(ans, candidates, selections, target, sum + candidates[i], step + 1, i);
        }
    }

    // 36
    // 有效的数独
    /*
        判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
        数字 1-9 在每一行只能出现一次。
        数字 1-9 在每一列只能出现一次。
        数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
    */
    bool isValidSudoku(vector<vector<char>> &board)
    {
        bool isValid = true;
        for (int i = 0; i < 9 && isValid; i++)
        {
            vector<bool> vis(9, false);
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] != '.')
                {
                    if (vis[board[i][j] - '0'])
                    {
                        isValid = false;
                        break;
                    }
                    vis[board[i][j] - '0'] = true;
                }
            }
        }
        for (int i = 0; i < 9 && isValid; i++)
        {
            vector<bool> vis(9, false);
            for (int j = 0; j < 9; j++)
            {
                if (board[j][i] != '.')
                {
                    if (vis[board[j][i] - '0'])
                    {
                        isValid = false;
                        break;
                    }
                    vis[board[j][i] - '0'] = true;
                }
            }
        }
        for (int i = 0; i < 3 && isValid; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int row = 3 * i;
                int col = 3 * j;
                vector<bool> vis(9, false);
                for (int r = row; r < row + 3; r++)
                {
                    for (int s = col; s < col + 3; s++)
                    {
                        if (board[r][s] != '.')
                        {
                            if (vis[board[r][s] - '0'])
                            {
                                isValid = false;
                                break;
                            }
                            vis[board[r][s] - '0'] = true;
                        }
                    }
                }
            }
        }
        return isValid;
    }

    // 34
    // 在排序数组中查找元素的第一个和最后一个位置
    /*
        给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
        你的算法时间复杂度必须是 O(log n) 级别。
        如果数组中不存在目标值，返回 [-1, -1]。
    */
    vector<int> searchRange(vector<int> &nums, int target)
    {
        int L = 0, R = nums.size() - 1;
        vector<int> ans(2, -1);
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            if (nums[mid] > target)
            {
                R = mid - 1;
            }
            else if (nums[mid] < target)
            {
                L = mid + 1;
            }
            else
            {
                ans[0] = mid;
                R = mid - 1;
            }
        }
        if (ans[0] != -1)
        {
            L = 0, R = nums.size() - 1;
            while (L <= R)
            {
                int mid = (L + R) >> 1;
                if (nums[mid] > target)
                {
                    R = mid - 1;
                }
                else if (nums[mid] < target)
                {
                    L = mid + 1;
                }
                else
                {
                    ans[1] = mid;
                    L = mid + 1;
                }
            }
        }
        return ans;
    }

    // 33
    // 搜索旋转排序数组
    /*
        假设按照升序排序的数组在预先未知的某个点上进行了旋转。
        ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
        搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
        你可以假设数组中不存在重复的元素。
        你的算法时间复杂度必须是 O(log n) 级别。
    */
    // 提示：不存在重复元素，注意有序情况否则分左半区和右半区.
    // 如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的
    int search(vector<int> &nums, int target)
    {
        int L = 0, R = nums.size() - 1;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            if (nums[mid] == target)
            {
                return mid;
            }
            if (nums[L] <= nums[mid])
            {
                if (target >= nums[L] && target < nums[mid])
                {
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }
            else
            {
                if (target > nums[mid] && target <= nums[R])
                {
                    L = mid + 1;
                }
                else
                {
                    R = mid - 1;
                }
            }
        }
        return -1;

        //        int len = nums.size();
        //        if (len == 0)
        //        {
        //            return -1;
        //        }
        //        int L = 0, R = len - 1;
        //        while (L <= R)
        //        {
        //            if (nums[R] > nums[L])
        //            {
        //                while (L <= R)
        //                {
        //                    int mid = (L + R) >> 1;
        //                    if (nums[mid] == target)
        //                    {
        //                        return mid;
        //                    }
        //                    if (nums[mid] > target)
        //                    {
        //                        R = mid - 1;
        //                    }
        //                    else
        //                    {
        //                        L = mid + 1;
        //                    }
        //                }
        //                return -1;
        //            }
        //            int mid = (L + R) >> 1;
        //            if (nums[mid] == target)
        //            {
        //                return mid;
        //            }
        //            if (nums[mid] > nums[R])
        //            {
        //                if (target < nums[mid])
        //                {
        //                    if (target > nums[R])
        //                    {
        //                        R = mid - 1;
        //                    }
        //                    else
        //                    {
        //                        L = mid + 1;
        //                    }
        //                }
        //                else
        //                {
        //                    L = mid + 1;
        //                }
        //            }
        //            else
        //            {
        //                if (target < nums[mid])
        //                {
        //                    R = mid - 1;
        //                }
        //                else
        //                {
        //                    if (target >= nums[L])
        //                    {
        //                        R = mid - 1;
        //                    }
        //                    else
        //                    {
        //                        L = mid + 1;
        //                    }
        //                }
        //            }
        //        }
        //        return -1;
    }

    // 31
    // 下一个排列
    /*
        实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
        如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
        必须原地修改，只允许使用额外常数空间。
    */
    void nextPermutation(vector<int> &nums)
    {
        int len = nums.size();
        bool rev = true;
        for (int i = 1; i < len; i++)
        {
            if (nums[i] <= nums[i - 1])
            {
                continue;
            }
            else
            {
                rev = false;
            }
        }
        if (rev)
        {
            reverse(nums.begin(), nums.end());
            return;
        }
        for (int i = len - 1; i > 0; i--)
        {
            if (nums[i] <= nums[i - 1])
            {
                continue;
            }
            for (int j = len - 1; j >= 0; j--)
            {
                if (nums[j] > nums[i - 1])
                {
                    swap(nums[j], nums[i - 1]);
                    break;
                }
            }
            int L = i, R = len - 1;
            while (L < R)
            {
                swap(nums[L++], nums[R--]);
            }
            return;
        }
    }

    // 29
    // 两数相除
    /*
        给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
        返回被除数 dividend 除以除数 divisor 得到的商。
    */
    // 提示：方法1：二分，乘法用位运算计算；方法2：逼近法
    int divide(int dividend, int divisor)
    {
        long long ans = 0, div1 = dividend, div2 = divisor;
        if (div1 < 0)
        {
            div1 = -div1;
        }
        if (div2 < 0)
        {
            div2 = -div2;
        }
        while (div1 >= div2)
        {
            long long a = 1, b = div2;
            while (div1 >= div2)
            {
                b <<= 1;
                if (b > div1)
                {
                    break;
                }
                a <<= 1;
            }
            div1 -= (b >> 1);
            ans = ans + a;
        }
        ans = (dividend < 0) ^ (divisor < 0) ? -ans : ans;
        return (ans > INT_MAX || ans < INT_MIN) ? INT_MAX : ans;
    }

    // 24
    // 两两交换链表中的节点
    /*
        给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
        你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
    */
    // 提示：注意[], [1], [1, 2, 3]
    ListNode *swapPairs(ListNode *head)
    {
        if (head == 0 || head->next == 0)
        {
            return head;
        }
        ListNode *node1 = head, *node2 = head->next, *node3, *node4;
        ListNode *newHead = node2;
        while (node1 != 0 && node2 != 0)
        {
            node3 = node2->next;
            node4 = node3 == 0 ? 0 : node3->next;

            node1->next = node4 == 0 ? node3 : node4;
            node2->next = node1;

            node1 = node3;
            node2 = node4;
        }
        return newHead;
    }

    // 22
    // 括号生成
    /*
        给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
    */
    vector<string> generateParenthesis(int n)
    {
        vector<string> ans;
        DFS_generateParenthesis(ans, "", 0, 0, n);
        return ans;
    }
    void DFS_generateParenthesis(vector<string> &ans, string s, int left, int length, int n)
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
        DFS_generateParenthesis(ans, s + "(", left + 1, length + 1, n);
        DFS_generateParenthesis(ans, s + ")", left - 1, length + 1, n);
    }

    // 19
    // 删除链表的倒数第N个节点
    /*
        给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
    */
    ListNode *removeNthFromEnd(ListNode *head, int n)
    {
        ListNode *node = head;
        int len = 0, k = 1;
        while (node != 0)
        {
            len++;
            node = node->next;
        }
        if (len == n)
        {
            return head->next;
        }
        node = head;
        while (true)
        {
            if (k == len - n)
            {
                node->next = node->next->next;
                return head;
            }
            node = node->next;
            k++;
        }
    }

    // 18
    // 四数之和
    /*
        给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
    */
    // 提示：固定两个数，双指针找另外两个；保存两两之和，再双指针
    vector<vector<int>> fourSum(vector<int> &nums, int target)
    {
        vector<vector<int>> ans;
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
                        if (!(ansSize > 0 && ans[ansSize - 1][0] == nums[i] && ans[ansSize - 1][1] == nums[j] && ans[ansSize - 1][2] == nums[L] && ans[ansSize - 1][3] == nums[R]))
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

    // 17
    // 电话号码的字母组合
    /*
        给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
        给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
    */
    vector<string> letterCombinations(string digits)
    {
        string nums[] = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> ans;
        DFS_letterCombinations(nums, ans, digits, 0, "");
        return ans;
    }
    void DFS_letterCombinations(string nums[], vector<string> &ans, string digits, int step, string num)
    {
        if (digits.size() == step)
        {
            if (step != 0)
            {
                ans.push_back(num);
            }
            return;
        }
        for (int i = 0; i < nums[digits[step] - '2'].size(); i++)
        {
            DFS_letterCombinations(nums, ans, digits, step + 1, num + nums[digits[step] - '2'][i]);
        }
    }

    // 16
    // 最接近的三数之和
    /*
        给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
    */
    int threeSumClosest(vector<int> &nums, int target)
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
    vector<vector<int>> threeSum(vector<int> &nums)
    {
        vector<vector<int>> ans;
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
                    if (!(ansSize > 0 && nums[i] == ans[ansSize - 1][0] && nums[L] == ans[ansSize - 1][1] && nums[R] == ans[ansSize - 1][2]))
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

    // 12
    // 整数转罗马数字
    /*
        罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
        字符          数值
        I             1
        V             5
        X             10
        L             50
        C             100
        D             500
        M             1000
        例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
        通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
        I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
        X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
        C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
        给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。
    */
    string intToRoman(int num)
    {
        int values[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        string reps[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        string ans = "";
        for (int i = 0; i < 13; i++)
        {
            while (num >= values[i])
            {
                ans = ans + reps[i];
                num -= values[i];
                if (num == 0)
                {
                    break;
                }
            }
        }
        return ans;
    }

    // 11
    // 盛最多水的容器
    /*
        给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
        说明：你不能倾斜容器，且 n 的值至少为 2。
    */
    // 提示：双指针法，矮的一方移动
    int maxArea(vector<int> &height)
    {
        int L = 0, R = height.size() - 1;
        int ans = 0;
        while (L < R)
        {
            ans = max(ans, min(height[L], height[R]) * (R - L));
            if (height[L] < height[R])
            {
                L++;
            }
            else
            {
                R--;
            }
        }
        return ans;
    }

    // 8
    // 字符串转整数
    /*
        当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
        该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
        注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
        在任何情况下，若函数不能进行有效的转换时，请返回 0。
    */
    int myAtoi(string s)
    {
        int len = s.size();
        for (int i = 0; i < len; i++)
        {
            if (s[i] != ' ')
            {
                if (s[i] == '+' || s[i] == '-' || (s[i] >= '0' && s[i] <= '9'))
                {
                    int index = i, sign = 1;
                    if (s[i] == '+')
                    {
                        index++;
                    }
                    else if (s[i] == '-')
                    {
                        index++;
                        sign = -1;
                    }
                    long long ans = 0;
                    while (index < len && s[index] >= '0' && s[index] <= '9')
                    {
                        ans = ans * 10 + s[index] - '0';
                        if (ans > INT_MAX)
                        {
                            break;
                        }
                        index++;
                    }
                    ans *= sign;
                    if (ans > INT_MAX)
                    {
                        return INT_MAX;
                    }
                    if (ans < INT_MIN)
                    {
                        return INT_MIN;
                    }
                    return ans;
                }
                else
                {
                    return 0;
                }
            }
        }
        return 0;
    }

    // 6
    // Z字形变换
    /*
        将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
        比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：
        L   C   I   R
        E T O E S I I G
        E   D   H   N
    */
    string convert(string s, int numRows)
    {
        if (numRows == 1)
        {
            return s;
        }
        int len = s.size(), index = 0, dir = 1;
        vector<string> str(numRows);
        for (int i = 0; i < len; i++)
        {
            str[index] += s[i];
            index += dir;
            if (index == numRows)
            {
                dir = -1;
                index = numRows - 2;
            }
            else if (index == -1)
            {
                dir = 1;
                index = 1;
            }
        }
        string ans = "";
        for (int i = 0; i < numRows; i++)
        {
            ans = ans + str[i];
        }
        return ans;
    }

    // 5
    // 最长回文子串
    /*
        给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
    */
    // Manacher算法O(N)时间复杂度
    string longestPalindrome(string s)
    {
        // O（n）算法
        int len = s.size();
        string str = "#";
        for (int i = 0; i < len; i++)
        {
            str = str + s[i] + "#";
        }
        len = str.size();
        vector<int> p(len);
        p[0] = 1;
        int mx = 1, id = 0;
        for (int i = 1; i < len; i++)
        {
            if (i < mx)
            {
                p[i] = min(mx - i, p[id * 2 - i]);
            }
            else
            {
                p[i] = 1;
            }
            while (i + p[i] < len && i - p[i] >= 0 && str[i + p[i]] == str[i - p[i]])
            {
                p[i]++;
            }
            if (p[i] + i > mx)
            {
                mx = i + p[i];
                id = i;
            }
        }
        int ma = 0;
        for (int i = 0; i < len; i++)
        {
            if (p[i] > ma)
            {
                ma = p[i];
                id = i;
            }
        }
        string ans = "";
        for (int i = id - p[id] + 1; i < id + p[id]; i++)
        {
            if (str[i] != '#')
            {
                ans += str[i];
            }
        }
        return ans;

        // 暴力算法
        /*
        int len = s.size(), ans = 0;
        string str;
        for (int i = 0; i < len; i++)
        {
            if (len - i < ans)
            {
                break;
            }
            for (int j = len - 1; j >= i; j--)
            {
                if (s[i] == s[j])
                {
                    int L = i, R = j;
                    while (L < R)
                    {
                        if (s[L] != s[R])
                        {
                            break;
                        }
                        L++;
                        R--;
                    }
                    if (L >= R && j - i + 1 > ans)
                    {
                        ans = j - i + 1;
                        str = "";
                        for (int r = i; r <= j; r++)
                        {
                            str += s[r];
                        }
                        break;
                    }
                }
            }
        }
        return str;
        */
    }

    // 3
    // 无重复字符的最长子串
    /*
        给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
    */
    // 提示：注意判断(c[s[i]] != -1 && c[s[i]] >= start)
    int lengthOfLongestSubstring(string s)
    {
        int len = s.size();
        int c[256], ans = 0, start = 0;
        memset(c, -1, sizeof(c));
        for (int i = 0; i < len; i++)
        {
            if (c[s[i]] != -1 && c[s[i]] >= start)
            {
                ans = max(ans, i - start);
                start = c[s[i]] + 1;
            }
            c[s[i]] = i;
        }
        ans = max(ans, len - start);
        return ans;
    }

    // 2
    // 两数相加
    /*
        给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
        如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
        您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
    */
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2)
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
};
int main()
{
    Solution *solution = new Solution();
    return 0;
}
