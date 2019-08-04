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
    // 45
    // 跳跃游戏II
    /*
        给定一个非负整数数组，你最初位于数组的第一个位置。
        数组中的每个元素代表你在该位置可以跳跃的最大长度。
        你的目标是使用最少的跳跃次数到达数组的最后一个位置。
        说明：
            假设你总是可以到达数组的最后一个位置。
    */
    int jump(vector<int>& nums)
    {
        int len = nums.size();
        int ans = 0, ma = 0;
        for (int i = 0; i < len; i++)
        {
            if (i + nums[i] > ma)
            {
                ma = i + nums[i];
                ans++;
                if (ma >= len)
                {
                    return ans;
                }
            }
            if (i > ma)
            {
                return -1;
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
    bool isMatch(string s, string p)
    {
        int len1 = s.size(), len2 = p.size();

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
