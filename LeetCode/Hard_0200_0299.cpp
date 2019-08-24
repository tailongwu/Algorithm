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
    // 239
    // 滑动窗口最大值
    /*
        给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
        返回滑动窗口中的最大值。
    */
    // 提示：双端队列维护递减有序列表。时间复杂度O(Nk)
    vector<int> maxSlidingWindow(vector<int> &nums, int k)
    {
        int len = nums.size();
        if (len < k || len == 0)
        {
            vector<int> result;
            return result;
        }
        vector<int> ans(len + 1 - k);
        deque<int> de;
        int left = -1;
        for (int i = 0; i < len; i++)
        {
            while (!de.empty() && de.back() < nums[i])
            {
                de.pop_back();
            }
            de.push_back(nums[i]);
            if (left != -1 && de.front() == nums[left])
            {
                de.pop_front();
            }
            if (i >= k - 1)
            {
                ans[++left] = de.front();
            }
        }
        return ans;
    }

    // 233
    // 数字1的个数
    /*
        给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
    */
    int countDigitOne(int n)
    {
        long long ans = 0;
        long long digitNum = 0, m = n, f[15];
        f[0] = 1;
        while (m != 0)
        {
            digitNum++;
            f[digitNum] = f[digitNum - 1] * 10;
            m /= 10;
        }
        for (int i = 0; i < digitNum; i++)
        {
            int left, right, cur;
            left = n / f[digitNum - i];
            right = n % f[digitNum - i - 1];
            cur = n % f[digitNum - i] / f[digitNum - i - 1];
            if (cur > 1)
            {
                ans += (left + 1) * f[digitNum - i - 1];
            }
            else if (cur == 1)
            {
                ans += left * f[digitNum - i - 1] + right + 1;
            }
            else
            {
                ans += left * f[digitNum - i - 1];
            }
        }
        return ans;
    }
};
int main()
{
    Solution *solution = new Solution();
    return 0;
}
