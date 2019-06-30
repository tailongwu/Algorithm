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
    // 199
    // 二叉树的右视图
    /*
        给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
    */
    vector<int> rightSideView(TreeNode* root)
    {
        vector<int> ans;
        DFS_levelOrder(root, ans, 0);
        return ans;
    }
    void DFS_levelOrder(TreeNode *node, vector<int> &ans, int step)
    {
        if (node == 0)
        {
            return;
        }
        if (ans.size() == step)
        {
            ans.push_back(node->val);
        }
        else
        {
            ans[step] = node->val;
        }
        DFS_levelOrder(node->left, ans, step + 1);
        DFS_levelOrder(node->right, ans, step + 1);
    }


    // 187
    // 重复的DNA序列
    /*
        所有 DNA 由一系列缩写为 A，C，G 和 T 的核苷酸组成，例如：“ACGAATTCCG”。在研究 DNA 时，识别 DNA 中的重复序列有时会对研究非常有帮助。
        编写一个函数来查找 DNA 分子中所有出现超过一次的10个字母长的序列（子串）。
    */
    vector<string> findRepeatedDnaSequences(string s)
    {
        int len = s.size();
        map<string, int> m;
        vector<string> ans;
        for (int i = 9; i < len; i++)
        {
            string str = "";
            for (int j = i - 9; j <= i; j++)
            {
                str += s[j];
            }
            m[str]++;
            if (m[str] == 2)
            {
                ans.push_back(str);
            }
        }
        return ans;
    }


    // 179
    // 最大数
    /*
        给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。
    */
    // 注意：全是0
    static bool cmp_largestNumber(const string &s1, const string &s2)
    {
        return s1 + s2 > s2 + s1;
    }
    string largestNumber(vector<int>& nums)
    {
        int len = nums.size();
        vector<string> num(len);
        for (int i = 0; i < len; i++)
        {
            string s = "";
            if (nums[i] == 0)
            {
                s = "0";
            }
            else
            {
                while (nums[i] != 0)
                {
                    s = s + (char)(nums[i] % 10 + '0');
                    nums[i] /= 10;
                }
            }
            int L = 0, R = s.size() - 1;
            while (L < R)
            {
                swap(s[L], s[R]);
                L++;
                R--;
            }
            num[i] = s;
        }

        string ans = "";
        sort(num.begin(), num.end(), cmp_largestNumber);
        for (int i = 0; i < num.size(); i++)
        {
            ans += num[i];
        }
        if (ans[0] == '0')
        {
            return "0";
        }
        return ans;
    }


    // 173
    // 二叉搜索树地带其
    /*
        实现一个二叉搜索树迭代器。你将使用二叉搜索树的根节点初始化迭代器。
        调用 next() 将返回二叉搜索树中的下一个最小的数。
    */
    // 提示：将所有当前根节点的左节点入栈
    stack<TreeNode*> sta_BSTIterator;
    void BSTIterator(TreeNode* root)
    {
        TreeNode *node = root;
        while (node != 0)
        {
            sta_BSTIterator.push(node);
            node = node->left;
        }
    }
    /** @return the next smallest number */
    int next_BSTIterator()
    {
        TreeNode *next = sta_BSTIterator.top();
        TreeNode *node = next->right;
        sta_BSTIterator.pop();
        while (node != 0)
        {
            sta_BSTIterator.push(node);
            node = node->left;
        }
        return next->val;
    }
    /** @return whether we have a next smallest number */
    bool hasNext_BSTIterator()
    {
        return !sta_BSTIterator.empty();
    }


    // 166
    // 分数到小数
    /*
        给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以字符串形式返回小数。
        如果小数部分为循环小数，则将循环的部分括在括号内。
    */
    string fractionToDecimal(int numerator, int denominator)
    {
        string ans = "";
        map<int, int> m;
        while (true)
        {
           // if (numerator)
        }
    }


    // 165
    // 比较版本号
    /*
        比较两个版本号 version1 和 version2。
        如果 version1 > version2 返回 1，如果 version1 < version2 返回 -1， 除此之外返回 0。
        你可以假设版本字符串非空，并且只包含数字和 . 字符。
         . 字符不代表小数点，而是用于分隔数字序列。
        例如，2.5 不是“两个半”，也不是“差一半到三”，而是第二版中的第五个小版本。
        你可以假设版本号的每一级的默认修订版号为 0。例如，版本号 3.4 的第一级（大版本）和第二级（小版本）修订号分别为 3 和 4。其第三级和第四级修订号均为 0。
    */
    int compareVersion(string version1, string version2)
    {
        int len1 = version1.size(), len2 = version2.size();
        int index1 = 0, index2 = 0, num1 = 0, num2 = 0;
        while (index1 < len1 || index2 < len2)
        {
            num1 = 0;
            num2 = 0;
            if (index1 == len1)
            {
                num1 = 0;
            }
            else if (version1[index1] != '.')
            {
                while (index1 != len1 && version1[index1] != '.')
                {
                    num1 = num1 * 10 + version1[index1] - '0';
                    index1++;
                }
            }
            else
            {
                index1++;
            }

            if (index2 == len2)
            {
                num2 = 0;
            }
            else if (version2[index2] != '.')
            {
                while (index2 != len2 && version2[index2] != '.')
                {
                    num2 = num2 * 10 + version2[index2] - '0';
                    index2++;
                }
            }
            else
            {
                index2++;
            }
            if (num1 > num2)
            {
                return 1;
            }
            if (num1 < num2)
            {
                return -1;
            }
        }
        return 0;
    }


    // 162
    // 寻找峰值
    /*
        峰值元素是指其值大于左右相邻值的元素。
        给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。
        数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。
        你可以假设 nums[-1] = nums[n] = -∞。
        说明：
        你的解法应该是 O(logN) 时间复杂度的。
    */
    // 提示：二分，比较mid和mid+1，如果nums[mid]<nums[mid+1]，右边一定存在
    // 注意：num可能为int最小值
    int findPeakElement(vector<int>& nums)
    {
        int L = 0, R = nums.size() - 1, len = nums.size();
        long long mi = -10e15;
        while (L <= R)
        {
            int mid = (L + R) >> 1;
            long long left = mid > 0 ? nums[mid - 1] : mi;
            long long right = mid < len - 1 ? nums[mid + 1] : mi;
            if (nums[mid] > left && nums[mid] > right)
            {
                return mid;
            }
            if (nums[mid] < right)
            {
                L = mid + 1;
            }
            else
            {
                R = mid;
            }
        }
        return 0;
    }


    // 153
    // 寻找旋转排序数组中的最小值
    /*
        假设按照升序排序的数组在预先未知的某个点上进行了旋转。
        ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
        请找出其中最小的元素。
        你可以假设数组中不存在重复元素。
    */
    // 提示：有四种情况
    int findMin(vector<int>& nums)
    {
        int L = 0, R = nums.size() - 1;
        int ans = nums[L], mid;
        while (L <= R)
        {
            mid = (L + R) >> 1;
            if (nums[mid] >= nums[L])
            {
                if (nums[mid] >= nums[R])
                {
                    L = mid + 1;
                }
                else
                {
                    R = mid - 1;
                }
            }
            else
            {
                if (nums[mid] >= nums[R])
                {
                    L = mid + 1;
                }
                else
                {
                    R = mid - 1;
                }
            }
            ans = min(ans, nums[mid]);
        }
        return ans;
    }


    // 152
    // 乘积最大子序列
    /*
        给定一个整数数组 nums ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。
    */
    // 提示：不能用min(nums[i],mi)，会出现断裂
    int maxProduct(vector<int>& nums)
    {
        int len = nums.size();
        int mi = 1, ma = 1, x, y, ans = INT_MIN;
        bool haveAns = false;
        for (int i = 0; i < len; i++)
        {
            x = min(nums[i], min(mi * nums[i], ma * nums[i]));
            y = max(nums[i], max(mi * nums[i], ma * nums[i]));
            mi = x;
            ma = y;
            ans = max(ans, ma);
        }
        return ans;
    }


    // 151
    // 翻转字符串里的单词
    /*
        给定一个字符串，逐个翻转字符串中的每个单词。
        说明：
        无空格字符构成一个单词。
        输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
        如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
    */
    string reverseWords(string s)
    {
        int len = s.size();
        string ans = "";
        for (int i = 0; i < len; i++)
        {
            if (s[i] != ' ')
            {
                if (ans.size() != 0)
                {
                    ans += ' ';
                }
                int j = i;
                while (j < len && s[j] != ' ')
                {
                    j++;
                }
                for (int k = j - 1; k >= i; k--)
                {
                    ans += s[k];
                }
                i = j;
            }
        }
        int L = 0, R = ans.size() - 1;
        while (L < R)
        {
            swap(ans[L], ans[R]);
            L++;
            R--;
        }
        return ans;

//        stack<string> sta;
//        int len = s.size();
//        for (int i = 0; i < len; i++)
//        {
//            if (s[i] != ' ')
//            {
//                string str = "";
//                while (i < len && s[i] != ' ')
//                {
//                    str += s[i];
//                    i++;
//                }
//                sta.push(str);
//            }
//        }
//        string ans = "";
//        while (!sta.empty())
//        {
//            ans = ans + sta.top();
//            sta.pop();
//            if (!sta.empty())
//            {
//                ans += " ";
//            }
//        }
//        return ans;
    }


    // 150
    // 逆波兰表达式求值
    /*
        根据逆波兰表示法，求表达式的值。
        有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
        说明：
        整数除法只保留整数部分。
        给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
    */
    // 提示：中缀转后缀
    /*
        从头到尾一遍扫描字符串：遇到运算数，则直接压入表达式结果栈；遇到运算符，则要根据运算符优先级分情况处理。
        运算符情况：1.左括号：直接压入符号栈。
                    2.加号、减号，优先级最低，所以要将栈中的加减乘除号先出栈到表达式结果栈，再将加减号入栈。
        　　　　　　3.乘号、除号，优先级最高，所以只需将栈中的乘除号出栈到表达式结果栈，再将此次的乘除号入栈。
        　　　　　　4.右括号：将栈中左括号之后入栈的运算符全部出栈到表达式结果栈，左括号出栈。
    */
    int evalRPN(vector<string>& tokens)
    {
        int len = tokens.size();
        stack<int> nums;
        for (int i = 0; i < len; i++)
        {
            if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/")
            {
                int num2 = nums.top();
                nums.pop();
                int num1 = nums.top();
                nums.pop();
                nums.push(Cal_evalRPN(tokens[i], num1, num2));
            }
            else
            {
                int num = 0;
                int sta = 0;
                if (tokens[i][0] == '-')
                {
                    sta++;
                }
                for (int j = sta; j < tokens[i].size(); j++)
                {
                    num = num * 10 + tokens[i][j] - '0';
                }
                if (tokens[i][0] == '-')
                {
                    num = -num;
                }
                nums.push(num);
            }
        }
        return nums.top();
    }
    int Cal_evalRPN(string op, int num1, int num2)
    {
        if (op == "+")
        {
            return num1 + num2;
        }
        else if (op == "-")
        {
            return num1 - num2;
        }
        else if (op == "*")
        {
            return num1 * num2;
        }
        else if (op == "/")
        {
            return num1 / num2;
        }
        return 0;
    }


    // 148
    // 排序链表
    /*
        在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
    */
    ListNode* sortList(ListNode* head)
    {
        // 归并排序
        return Merge_sortList(head);
    }
    ListNode *Merge_sortList(ListNode *node)
    {
        if (node == 0 || node->next == 0)
        {
            return node;
        }
        ListNode *node1 = node, *node2 = node, *last;
        while (node2 != 0)
        {
            last = node1;
            node1 = node1->next;
            node2 = node2->next == 0 ? 0 : node2->next->next;
        }
        last->next = 0;
        node2 = Merge_sortList(node);
        node1 = Merge_sortList(node1);
        ListNode *newHead = 0;
        while (node1 != 0 && node2 != 0)
        {
            if (node1->val < node2->val)
            {
                if (newHead == 0)
                {
                    newHead = node1;
                    node = newHead;
                    node1 = node1->next;
                }
                else
                {
                    node->next = node1;
                    node1 = node1->next;
                    node = node->next;
                }
            }
            else
            {
                if (newHead == 0)
                {
                    newHead = node2;
                    node = newHead;
                    node2 = node2->next;
                }
                else
                {
                    node->next = node2;
                    node2 = node2->next;
                    node = node->next;
                }
            }
        }
        if (node != 0)
        {
            node->next = node1 == 0 ? node2 : node1;
        }
        return newHead;
    }


    // 147
    // 对链表进行插入排序
    /*
        对链表进行插入排序。
    */
    // 注意：相等情况[1,1]
    ListNode* insertionSortList(ListNode* head)
    {
        ListNode *newHead = 0, *node = head, *newNode, *last, *next;
        while (node != 0)
        {
            if (newHead == 0)
            {
                newHead = node;
                node = node->next;
                newHead->next = 0;
            }
            else
            {
                if (node->val <= newHead->val)
                {
                    newNode = node;
                    node = node->next;
                    newNode->next = newHead;
                    newHead = newNode;
                }
                else
                {
                    newNode = newHead;
                    while (newNode != 0 && newNode->val < node->val)
                    {
                        last = newNode;
                        newNode = newNode->next;
                    }
                    last->next = node;
                    next = node->next;
                    node->next = newNode;
                    node = next;
                }
            }
        }
        return newHead;
    }


    // 144
    // 二叉树的前序遍历
    /*
        给定一个二叉树，返回它的 前序 遍历。
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


    // 143
    // 重排链表
    /*
        给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
        将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
        你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
    */
    // 提示：分割，反转，合并
    void reorderList(ListNode* head)
    {
        if (head == 0)
        {
            return;
        }
        ListNode *last, *node, *node1, *node2, *next, *head1, *head2;
        node1 = head;
        node2 = head;
        while (node2 != 0)
        {
            last = node1;
            node1 = node1->next;
            node2 = node2->next == 0 ? 0 : node2->next->next;
        }
        head1 = head;
        last->next = 0;
        head2 = node1;

        node2 = head2;
        head2 = 0;
        while (node2 != 0)
        {
            next = node2->next;
            node2->next = head2;
            head2 = node2;
            node2 = next;
        }

        head = head1;
        node = head;
        node1 = head1->next;
        node2 = head2;
        while (node1 != 0 && node2 != 0)
        {
            node->next = node2;
            node2 = node2->next;
            node = node->next;
            if (node != 0)
            {
                node->next = node1;
                node1 = node1->next;
                node = node->next;
            }
        }
        if (node != 0)
        {
            node->next = node1 == 0 ? node2 : node1;
            node = node->next;
        }
        if (node != 0)
        {
            node->next = 0;
        }
    }


    // 142
    // 环形链表 II
    /*
        给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
        为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
        说明：不允许修改给定的链表。
    */
    ListNode *detectCycle(ListNode *head)
    {
        if (head == 0)
        {
            return 0;
        }
        ListNode *node1 = head, *node2 = head;
        while (node1 != 0 && node2 != 0)
        {
            node1 = node1->next;
            node2 = node2->next == 0 ? 0 : node2->next->next;
            if (node1 == node2)
            {
                break;
            }
        }
        if (node1 != node2 || node2 == 0)
        {
            return 0;
        }
        node2 = head;
        while (node1 != node2)
        {
            node1 = node1->next;
            node2 = node2->next;
        }
        return node1;
    }


    // 139
    // 单词拆分
    /*
        给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
        说明：
        拆分时可以重复使用字典中的单词。
        你可以假设字典中没有重复的单词
    */
    // 提示：动态规划或dfs
    bool wordBreak(string s, vector<string>& wordDict)
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


    // 138
    // 复制带随机指针的链表
    /*
        给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
        要求返回这个链表的深拷贝。
    */
    Node3* copyRandomList(Node3* head)
    {
        Node3 *node = head, *newNode, *next, *newNext, *newHead = 0;
        while (node != 0)
        {
            Node3 *newNode = new Node3(node->val, 0, 0);
            next = node->next;
            node->next = newNode;
            newNode->next = next;
            node = next;
        }

        node = head;
        while (node != 0)
        {
            node->next->random = node->random == 0 ? 0 : node->random->next;
            node = node->next->next;
        }

        node = head;
        if (node != 0)
        {
            newHead = node->next;
            newNode = newHead;
            while (node != 0)
            {
                newNext = newNode->next;
                node->next = newNext;
                newNode->next = newNext == 0 ? 0 : newNext->next;
                node = node->next;
                newNode = newNode->next;
            }
        }
        return newHead;
    }


    // 137
    // 只出现一次的数字 II
    /*
        给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
        说明：
        你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
    */
    // 方法1：每位为1的个数，个数如果模3为1，该数的该位则为1。
    // 方法2：想办法出现3次会抵消为0。
    int singleNumberII(vector<int>& nums)
    {
        int len = nums.size(), ans = 0;
        for (int i = 0; i < 32; i++)
        {
            int cnt = 0;
            for (int j = 0; j < len; j++)
            {
                cnt = cnt + ((nums[j] >> i) & 1);
            }
            if (cnt % 3 != 0)
            {
                ans = ans + (1 << i);
            }
        }
        return ans;
    }


    // 134
    // 加油站
    /*
        在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
        你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
        如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
        说明:
        如果题目有解，该答案即为唯一答案。
        输入数组均为非空数组，且长度相同。
        输入数组中的元素均为非负数。
    */
    // 提示：复制长度，和一直大于等于0
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost)
    {
        int len = gas.size();
        vector<int> gas2(len << 1);
        vector<int> cost2(len << 1);
        for (int i = 0; i < len; i++)
        {
            gas2[i] = gas[i];
            gas2[i + len] = gas[i];
            cost2[i] = cost[i];
            cost2[i + len] = cost[i];
        }
        int remain = 0, sta = -1, k = 0;
        for (int i = 0; i < len * 2; i++)
        {
            remain += gas2[i] - cost2[i];
            k++;
            if (remain < 0)
            {
                k = 0;
                sta = i + 1;
                remain = 0;
            }
            if (k == len)
            {
                break;
            }
        }
        if (k != len)
        {
            return -1;
        }
        return sta;
    }


    // 133
    // 克隆图
    /*
        给定无向连通图中一个节点的引用，返回该图的深拷贝（克隆）。图中的每个节点都包含它的值 val（Int） 和其邻居的列表（list[Node]）。
        提示：
        节点数介于 1 到 100 之间。
        无向图是一个简单图，这意味着图中没有重复的边，也没有自环。
        由于图是无向的，如果节点 p 是节点 q 的邻居，那么节点 q 也必须是节点 p 的邻居。
        必须将给定节点的拷贝作为对克隆图的引用返回。
    */
    map<int, Node2*> map_cloneGraph;
    Node2* cloneGraph(Node2* node)
    {
        if (node == 0)
        {
            return node;
        }
        vector<Node2*> neighbors;
        Node2 *copiedNode = new Node2(node->val, neighbors);
        map_cloneGraph[node->val] = copiedNode;
        for (int i = 0; i < node->neighbors.size(); i++)
        {
            if (map_cloneGraph[node->neighbors[i]->val] == 0)
            {
                Node2 *tmp = cloneGraph(node->neighbors[i]);
                map_cloneGraph[node->neighbors[i]->val] = tmp;
            }
            copiedNode->neighbors.push_back(map_cloneGraph[node->neighbors[i]->val]);
        }
        return copiedNode;
    }


    // 131
    // 分割回文串
    /*
        给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
        返回 s 所有可能的分割方案。
    */
    // 注意：字符串可能很长；可以用map优化判断是否未回文串
    vector<vector<string> > partition(string s)
    {
        int len = s.size();
        vector<vector<string> > ans;
        vector<string> result(len);
        DFS_partition(ans, result, s, len, 0, 0);
        return ans;
    }
    void DFS_partition(vector<vector<string> > &ans, vector<string> &result, string &s, int len, int sta, int stp)
    {
        if (sta == len)
        {
            vector<string> a(stp);
            for (int i = 0; i < stp; i++)
            {
                a[i] = result[i];
            }
            ans.push_back(a);
            return;
        }
        for (int i = sta; i < len; i++)
        {
            int L = sta, R = i;
            while (L < R && s[L] == s[R])
            {
                L++;
                R--;
            }
            if (L >= R)
            {
                string str = "";
                for (int j = sta; j <= i; j++)
                {
                    str += s[j];
                }
                result[stp] = str;
                DFS_partition(ans, result, s, len, i + 1, stp + 1);
            }
        }
    }


    // 130
    // 被围绕的区域
    /*
        给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
        找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
        被围绕的区间不会存在于边界上。
    */
    void solve(vector<vector<char> >& board)
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
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (board[i][j] == 'O')
                {
                    bool around = true;
                    DFS_solve(board, row, col, i, j, around, 'O', 'P');
                    if (around)
                    {
                        DFS_solve(board, row, col, i, j, around, 'P', 'X');
                    }
                }
            }
        }
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (board[i][j] =='P')
                {
                    board[i][j] = 'O';
                }
            }
        }
    }
    void DFS_solve(vector<vector<char> >& board, int row, int col, int x, int y, bool &around, char s, char d)
    {
        if (x >= 0 && x < row && y >= 0 && y < col)
        {
            if (board[x][y] == s)
            {
                if (x == 0 || y == 0 || x == row - 1 || y == col - 1)
                {
                    around = false;
                }
                board[x][y] = d;
            }
            else
            {
                return;
            }
        }
        else
        {
            return;
        }
        int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int i = 0; i < 4; i++)
        {
            int xx = x + dir[i][0];
            int yy = y + dir[i][1];
            DFS_solve(board, row, col, xx, yy, around, s, d);
        }
    }


    // 129
    // 求根到叶子节点数字之和
    /*
        给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
        例如，从根到叶子节点路径 1->2->3 代表数字 123。
        计算从根到叶子节点生成的所有数字之和。
        说明: 叶子节点是指没有子节点的节点。
    */
    int sumNumbers(TreeNode* root)
    {
        int sum = 0;
        vector<int> path;
        DFS_sumNumbers(root, path, 0, sum);
        return sum;
    }
    void DFS_sumNumbers(TreeNode *node, vector<int> path, int stp, int &sum)
    {
        if (node == 0)
        {
            return;
        }
        if (path.size() == stp)
        {
            path.push_back(node->val);
        }
        else
        {
            path[stp] = node->val;
        }
        stp++;

        if (node->left == 0 && node->right == 0)
        {
            int num = 0;
            for (int i = 0; i < stp; i++)
            {
                num = num * 10 + path[i];
            }
            sum += num;
            return;
        }
        DFS_sumNumbers(node->left, path, stp, sum);
        DFS_sumNumbers(node->right, path, stp, sum);
    }


    // 127
    // 单词接龙
    /*
        给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：
        每次转换只能改变一个字母。
        转换过程中的中间单词必须是字典中的单词。
        说明:
        如果不存在这样的转换序列，返回 0。
        所有单词具有相同的长度。
        所有单词只由小写字母组成。
        字典中不存在重复的单词。
        你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
    */
    // 提示：也可双端bfs
    struct P_ladderLength
    {
        string word;
        int stp;
    };
    int ladderLength(string beginWord, string endWord, vector<string>& wordList)
    {
        int wordListSize = wordList.size();
        int wordSize = beginWord.size();
        vector<bool> vis(wordListSize);
        bool exist = false;
        for (int i = 0; i < wordListSize; i++)
        {
            if (beginWord == wordList[i])
            {
                vis[i] = true;
            }
            if (endWord == wordList[i])
            {
                exist = true;
            }
        }
        if (!exist)
        {
            return 0;
        }

        queue<P_ladderLength> Q;
        P_ladderLength p, q;
        p.word = beginWord;
        p.stp = 0;
        Q.push(p);
        while (!Q.empty())
        {
            p = Q.front();
            Q.pop();
            if (p.word == endWord)
            {
                return p.stp + 1;
            }
            for (int i = 0; i < wordListSize; i++)
            {
                if (!vis[i])
                {
                    int diff = 0;
                    for (int j = 0; j < wordSize; j++)
                    {
                        if (p.word[j] != wordList[i][j])
                        {
                            diff++;
                            if (diff > 1)
                            {
                                break;
                            }
                        }
                    }
                    if (diff == 1)
                    {
                        q.word = wordList[i];
                        q.stp = p.stp + 1;
                        vis[i] = true;
                        Q.push(q);
                    }
                }
            }
        }
        return 0;
    }


    // 120
    // 三角形最小路径和
    /*
        给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
    */
    int minimumTotal(vector<vector<int> >& triangle)
    {
        int row = triangle.size();
        if (row == 0)
        {
            return 0;
        }
        for (int i = 1; i < row; i++)
        {
            triangle[i][0] += triangle[i - 1][0];
            triangle[i][i] += triangle[i - 1][i - 1];
            for (int j = 1; j < i; j++)
            {
                triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j]);
            }
        }
        int ans = triangle[row - 1][0];
        for (int i = 0; i < row; i++)
        {
            ans = min(ans, triangle[row - 1][i]);
        }
        return ans;
    }


    // 117
    // 填充每个节点的下一个右侧节点指针 II
    /*
        给定一个二叉树
        struct Node {
          int val;
          Node *left;
          Node *right;
          Node *next;
        }
        填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
        初始状态下，所有 next 指针都被设置为 NULL。
    */
    // 注意：未通过case#34
    Node* connect(Node* root)
    {
        if (root == 0)
        {
            return 0;
        }
        if (root->left != 0)
        {
            root->left->next = root->right;
        }
        if (root->next != 0)
        {
            Node *left = (root->right != 0 ? root->right : (root->left != 0 ? left : 0));
            Node *right = (root->next->left != 0 ? root->next->left : (root->next->right != 0 ? root->next->right : 0));
            if (left != 0)
            {
                left->next = right;
            }
        }
        connect(root->left);
        connect(root->right);
        return root;
    }


    // 116
    // 填充每个节点的下一个右侧节点指针
    /*
        给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
        struct Node {
          int val;
          Node *left;
          Node *right;
          Node *next;
        }
        填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
        初始状态下，所有 next 指针都被设置为 NULL。
    */
    Node* connect116(Node* root)
    {
        if (root == 0)
        {
            return 0;
        }
        if (root->left != 0)
        {
            root->left->next = root->right;
            if (root->next != 0)
            {
                root->right->next = root->next->left;
            }
            connect116(root->left);
            connect116(root->right);
        }
        return root;
    }


    // 114
    // 二叉树展开为链表
    /*
        给定一个二叉树，原地将它展开为链表。
    */
    // 后序遍历
    void flatten(TreeNode* root)
    {
        if (root == 0)
        {
            return;
        }
        flatten(root->left);
        flatten(root->right);
        TreeNode *right = root->right;
        root->right = root->left;
        root->left = 0;
        while (root->right != 0)
        {
            root = root->right;
        }
        root->right = right;
    }


    // 113
    // 路径总和
    /*
        给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
        说明: 叶子节点是指没有子节点的节点。
    */
    // 注意：有负数
    vector<vector<int> > pathSum(TreeNode* root, int sum)
    {
        vector<vector<int> > ans;
        vector<int> result;
        DFS_pathSum(ans, result, root, sum, 0, 0);
        return ans;
    }
    void DFS_pathSum(vector<vector<int> > &ans, vector<int> &result, TreeNode *node, int sum, int curSum, int stp)
    {
        if (node == 0)
        {
            return;
        }
        if (result.size() == stp)
        {
            result.push_back(node->val);
        }
        else
        {
            result[stp] = node->val;
        }
        stp++;
        if (curSum + node->val == sum && node->left == 0 && node->right == 0)
        {
            vector<int> a(stp);
            for (int i = 0; i < stp; i++)
            {
                a[i] = result[i];
            }
            ans.push_back(a);
            return;
        }
        DFS_pathSum(ans, result, node->left, sum, curSum + node->val, stp);
        DFS_pathSum(ans, result, node->right, sum, curSum + node->val, stp);
    }


    // 109
    // 有序链表转换二叉搜索树
    /*
        给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
        本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
    */
    TreeNode* sortedListToBST(ListNode* head)
    {
        vector<int> nodes;
        while (head != 0)
        {
            nodes.push_back(head->val);
            head = head->next;
        }
        return DFS_sortedListToBST(nodes, 0, nodes.size() - 1);
    }
    TreeNode* DFS_sortedListToBST(vector<int> &nodes, int sta, int en)
    {
        if (sta > en)
        {
            return 0;
        }
        int mid = (sta + en) >> 1;
        TreeNode *root = new TreeNode(nodes[mid]);
        root->left = DFS_sortedListToBST(nodes, sta, mid - 1);
        root->right = DFS_sortedListToBST(nodes, mid + 1, en);
        return root;
    }


    // 106
    // 从中序与后序遍历序列构造二叉树
    /*
        根据一棵树的中序遍历与后序遍历构造二叉树。
        注意:
        你可以假设树中没有重复的元素。
    */
    // 注意：先算右子树
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder)
    {
        int stp = postorder.size() - 1;
        return DFS_buildTree(inorder, postorder, stp, 0, inorder.size() - 1);
    }
    TreeNode* DFS_buildTree(vector<int>& inorder, vector<int>& postorder, int &stp, int sta, int en)
    {
        if (sta > en || stp >= postorder.size())
        {
            return 0;
        }
        int index = -1;
        for (int i = sta; i <= en; i++)
        {
            if (inorder[i] == postorder[stp])
            {
                index = i;
                break;
            }
        }
        TreeNode *root = new TreeNode(postorder[stp]);
        stp--;
        root->right = DFS_buildTree(inorder, postorder, stp, index + 1, en);
        root->left = DFS_buildTree(inorder, postorder, stp, sta, index - 1);
        return root;
    }


    // 105
    // 从前序与中序遍历序列构造二叉树
    /*
        根据一棵树的前序遍历与中序遍历构造二叉树。
        注意:
        你可以假设树中没有重复的元素。
    */
    // 注意：stp
    TreeNode* buildTree105(vector<int>& preorder, vector<int>& inorder)
    {
        int stp = 0;
        return DFS_buildTree105(preorder, inorder, stp, 0, inorder.size() - 1);
    }
    TreeNode* DFS_buildTree105(vector<int>& preorder, vector<int>& inorder, int &stp, int sta, int en)
    {
        if (sta > en)
        {
            return 0;
        }
        int index = -1;
        for (int i = sta; i <= en; i++)
        {
            if (inorder[i] == preorder[stp])
            {
                index = i;
                break;
            }
        }
        TreeNode *root = new TreeNode(preorder[stp]);
        stp++;
        root->left = DFS_buildTree105(preorder, inorder, stp, sta, index - 1);
        root->right = DFS_buildTree105(preorder, inorder, stp, index + 1, en);
        return root;
    }


    // 103
    // 二叉树的锯齿形层次遍历
    /*
        给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
    */
    vector<vector<int> > zigzagLevelOrder(TreeNode* root)
    {
        vector<vector<int> > ans;
        DFS_zigzagLevelOrder(root, ans, 0);
        int len = ans.size();
        for (int i = 1; i < len; i += 2)
        {
            int L = 0, R = ans[i].size() - 1;
            while (L < R)
            {
                swap(ans[i][L++], ans[i][R--]);
            }
        }
        return ans;
    }
    void DFS_zigzagLevelOrder(TreeNode *node, vector<vector<int> > &ans, int step)
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
        DFS_zigzagLevelOrder(node->left, ans, step + 1);
        DFS_zigzagLevelOrder(node->right, ans, step + 1);
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
};
int main()
{
    Solution* solution = new Solution();
    return 0;
}
