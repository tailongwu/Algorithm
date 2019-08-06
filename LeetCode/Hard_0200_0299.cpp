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
    Solution* solution = new Solution();
    return 0;
}
