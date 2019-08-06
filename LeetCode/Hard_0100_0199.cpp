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
    // 145
    // 二叉树的后序遍历
    /*
        给定一个二叉树，返回它的 后序 遍历。
        进阶: 递归算法很简单，你可以通过迭代算法完成吗？
    */
    vector<int> postorderTraversal(TreeNode* root)
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
        stack<TreeNode*> sta1;
        stack<TreeNode*> sta2;
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
};
int main()
{
    Solution* solution = new Solution();
    return 0;
}
