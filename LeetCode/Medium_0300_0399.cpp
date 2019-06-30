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
