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
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution
{
public:
    // 95
    // ��ͬ�Ķ���������II
    /*
        ����һ������ n������������ 1 ... n Ϊ�ڵ�����ɵĶ�����������
    */
    vector<TreeNode*> generateTrees(int n)
    {

    }


    // 94
    // ���������������
    /*
        ����һ���������������������� ������
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


    // 93
    // ��ԭIP��ַ
    /*
        ����һ��ֻ�������ֵ��ַ�������ԭ�����������п��ܵ� IP ��ַ��ʽ��
    */
    // ��ʾ��4��forѭ����dfs��
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
    // ��ת����II
    /*
        ��ת��λ�� m �� n ��������ʹ��һ��ɨ����ɷ�ת��
        ˵��:
        1 �� m �� n �� �����ȡ�
    */
    ListNode* reverseBetween(ListNode* head, int m, int n)
    {
        ListNode *last = 0, *node = head;
        for (int i = 0; i < m - 1; i++)
        {
            last = node;
            node = node->next;
        }
        ListNode *newHead, *cur, *right = node;
        for (int i = m;i <= n; i++)
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
    // ���뷽��
    /*
        һ��������ĸ A-Z ����Ϣͨ�����·�ʽ�����˱��룺
        'A' -> 1
        'B' -> 2
        ...
        'Z' -> 26
        ����һ��ֻ�������ֵķǿ��ַ������������뷽����������
    */
    // ��ʾ��dp����dfs��dpЧ�ʸ��ߣ�����ֻ��3������
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
                dp[i] += dp[i-1];
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
    // �Ӽ�II
    /*
        ����һ�����ܰ����ظ�Ԫ�ص��������� nums�����ظ��������п��ܵ��Ӽ����ݼ�����
        ˵�����⼯���ܰ����ظ����Ӽ���
    */
    vector<vector<int> > subsetsWithDup(vector<int>& nums)
    {
        sort(nums.begin(), nums.end());
        vector<vector<int> > ans;
        vector<int> result(nums.size());
        DFS_subsetsWithDup(nums, ans, result, nums.size(), 0, 0);
        return ans;
    }
    void DFS_subsetsWithDup(vector<int> &nums, vector<vector<int> > &ans, vector<int> &result, int len, int sta, int stp)
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
    // ���ױ���
    /*
        ���ױ�����һ������������ϵͳ���ڸ�ϵͳ�У�������������ֵ����һ��λ���Ĳ��졣
        ����һ�����������λ���ķǸ����� n����ӡ����ױ������С����ױ������б����� 0 ��ͷ��
    */
    // ��ʾ�����ױ������ɣ�֤��������
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
    // �ָ�����
    /*
        ����һ�������һ���ض�ֵ x����������зָ���ʹ������С�� x �Ľڵ㶼�ڴ��ڻ���� x �Ľڵ�֮ǰ��
        ��Ӧ����������������ÿ���ڵ�ĳ�ʼ���λ�á�
    */
    ListNode* partition(ListNode* head, int x)
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
    // ɾ�����������е��ظ�Ԫ�� II
    /*
        ����һ����������ɾ�����к����ظ����ֵĽڵ㣬ֻ����ԭʼ������ û���ظ����� �����֡�
    */
    ListNode* deleteDuplicates(ListNode* head)
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
    // ������ת�������� II
    /*
        ���谴�����������������Ԥ��δ֪��ĳ�����Ͻ�������ת��
        ( ���磬���� [0,0,1,2,2,5,6] ���ܱ�Ϊ [2,5,6,0,0,1,2] )��
        ��дһ���������жϸ�����Ŀ��ֵ�Ƿ�����������С������ڷ��� true�����򷵻� false��
    */
    bool searchII(vector<int>& nums, int target)
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
    // ɾ�����������е��ظ���II
    /*
        ����һ���������飬����Ҫ��ԭ��ɾ���ظ����ֵ�Ԫ�أ�ʹ��ÿ��Ԫ�����������Σ������Ƴ���������³��ȡ�
        ��Ҫʹ�ö��������ռ䣬�������ԭ���޸��������鲢��ʹ�� O(1) ����ռ����������ɡ�
    */
    int removeDuplicates(vector<int>& nums)
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
    // ��������
    /*
        ����һ����ά�����һ�����ʣ��ҳ��õ����Ƿ�����������С�
        ���ʱ��밴����ĸ˳��ͨ�����ڵĵ�Ԫ���ڵ���ĸ���ɣ����С����ڡ���Ԫ������Щˮƽ���ڻ�ֱ���ڵĵ�Ԫ��ͬһ����Ԫ���ڵ���ĸ�������ظ�ʹ�á�
    */
    bool exist(vector<vector<char> >& board, string word)
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
    void DFS_exist(vector<vector<char> >& board, string word, int row, int col, int len, int x, int y, int stp, bool &exist)
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
    // �Ӽ�
    /*
        ����һ�鲻���ظ�Ԫ�ص��������� nums�����ظ��������п��ܵ��Ӽ����ݼ�����
        ˵�����⼯���ܰ����ظ����Ӽ���
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


    // 77
    // ���
    /*
        ������������ n �� k������ 1 ... n �����п��ܵ� k ��������ϡ�
    */
    vector<vector<int> > combine(int n, int k)
    {
        vector<vector<int> > ans;
        vector<int> result(k);
        DFS_combine(ans, result, n, k, 1, 0);
        return ans;
    }
    void DFS_combine(vector<vector<int> > &ans, vector<int> &result, int n, int k, int sta, int step)
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
    // ��ɫ����
    /*
        ����һ��������ɫ����ɫ����ɫ��һ�� n ��Ԫ�ص����飬ԭ�ض����ǽ�������ʹ����ͬ��ɫ��Ԫ�����ڣ������պ�ɫ����ɫ����ɫ˳�����С�
        �����У�����ʹ������ 0�� 1 �� 2 �ֱ��ʾ��ɫ����ɫ����ɫ��
        ע��:
        ����ʹ�ô�����е����������������⡣
        ���ף�
        һ��ֱ�۵Ľ��������ʹ�ü������������ɨ���㷨��
        ���ȣ����������0��1 �� 2 Ԫ�صĸ�����Ȼ����0��1��2��������д��ǰ���顣
        �������һ����ʹ�ó����ռ��һ��ɨ���㷨��
    */
    // ��ʾ����ָ�뷨
    void sortColors(vector<int>& nums)
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


    // ������ά����
    // 74
    /*
        ��дһ����Ч���㷨���ж� m x n �����У��Ƿ����һ��Ŀ��ֵ���þ�������������ԣ�
        ÿ���е����������Ұ��������С�
        ÿ�еĵ�һ����������ǰһ�е����һ��������
    */
    // ��ʾ�������Ͻǿ�ʼ
    bool searchMatrix(vector<vector<int> >& matrix, int target)
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
    // ��������
    /*
        ����һ�� m x n �ľ������һ��Ԫ��Ϊ 0�����������к��е�����Ԫ�ض���Ϊ 0����ʹ��ԭ���㷨��ʹ�ó����ռ䡣
    */
    // ��ʾ���ҵ���һ��Ϊ0�ģ���ô������0�ֽ�ɵ�һ��0���ڵ��к��м�¼������ٱ�����
    void setZeroes(vector<vector<int> >& matrix)
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
    // ��·��
    /*
        �� Unix ������һ���ļ��ľ���·��������Ҫ���������߻��仰˵������ת��Ϊ�淶·����
        �� Unix �����ļ�ϵͳ�У�һ���㣨.����ʾ��ǰĿ¼�������⣬������ ��..�� ��ʾ��Ŀ¼�л�����һ����ָ��Ŀ¼�������߶������Ǹ������·������ɲ��֡�������Ϣ����ģ�Linux / Unix�еľ���·�� vs ���·��
        ��ע�⣬���صĹ淶·������ʼ����б�� / ��ͷ����������Ŀ¼��֮�����ֻ��һ��б�� /�����һ��Ŀ¼����������ڣ������� / ��β�����⣬�淶·�������Ǳ�ʾ����·��������ַ�����
    */
    // ��ʾ��ע���������ַ���ϻ��߳������������ļ���
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
                i= j - 1;
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
    // ��С·����
    /*
        ����һ�������Ǹ������� m x n �������ҳ�һ�������Ͻǵ����½ǵ�·����ʹ��·���ϵ������ܺ�Ϊ��С��
        ˵����ÿ��ֻ�����»��������ƶ�һ����
    */
    int minPathSum(vector<vector<int> >& grid)
    {
        int row = grid.size();
        int col = grid[0].size();
        if (row == 0 || col == 0)
        {
            return 0;
        }
        vector<vector<int> > dp(row);
        for (int i = 0; i < row; i++)
        {
            dp[i].resize(col);
        }
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
    // ��ͬ·��II
    /*
        һ��������λ��һ�� m x n ��������Ͻ� ����ʼ������ͼ�б��Ϊ��Start�� ����
        ������ÿ��ֻ�����»��������ƶ�һ������������ͼ�ﵽ��������½ǣ�����ͼ�б��Ϊ��Finish������
        ���ڿ������������ϰ����ô�����Ͻǵ����½ǽ����ж�������ͬ��·����
    */
    int uniquePathsWithObstacles(vector<vector<int> >& obstacleGrid)
    {
        int row = obstacleGrid.size();
        int col = obstacleGrid[0].size();
        if (row == 0 || col == 0 || obstacleGrid[0][0] == 1)
        {
            return 0;
        }
        vector<vector<long long> > dp(row);
        for (int i = 0; i < row; i++)
        {
            dp[i].resize(col);
        }
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
    // ��ͬ·��
    /*
        һ��������λ��һ�� m x n ��������Ͻ� ����ʼ������ͼ�б��Ϊ��Start�� ����
        ������ÿ��ֻ�����»��������ƶ�һ������������ͼ�ﵽ��������½ǣ�����ͼ�б��Ϊ��Finish������
        ���ܹ��ж�������ͬ��·����
        ˵����m �� n ��ֵ�������� 100��
    */
    // ��ʾ��ans=C(m+n-2,n-1)��ʵ����100��50��Խ�磬����û�и�������Ҳ���Էֽ���ʽ�����������ǡ�
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
    // ��ת����
    /*
        ����һ��������ת����������ÿ���ڵ������ƶ� k ��λ�ã����� k �ǷǸ�����
    */
    ListNode* rotateRight(ListNode* head, int k)
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
    // ��k������
    /*
        �������� [1,2,3,��,n]��������Ԫ�ع��� n! �����С����� n �� k�����ص� k �����С�
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
            while (k > f[n - 1 -i])
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
    // ��������II
    /*
        ����һ�������� n������һ������ 1 �� n2 ����Ԫ�أ���Ԫ�ذ�˳ʱ��˳���������е������ξ���
    */
    vector<vector<int> > generateMatrix(int n)
    {
        vector<vector<int> > ans(n);
        for (int i = 0; i < n; i++)
        {
            ans[i].resize(n);
        }
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
    // �ϲ�����
    /*
        ����һ������ļ��ϣ���ϲ������ص������䡣
    */
    vector<vector<int> > merge(vector<vector<int> >& intervals)
    {
        int len = intervals.size();
        vector<vector<int> > ans;
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
    // ��Ծ��Ϸ
    /*
        ����һ���Ǹ��������飬�����λ������ĵ�һ��λ�á�
        �����е�ÿ��Ԫ�ش������ڸ�λ�ÿ�����Ծ����󳤶ȡ�
        �ж����Ƿ��ܹ��������һ��λ�á�
    */
    bool canJump(vector<int>& nums)
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

// ����2
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
    // ��������
    /*
        ����һ������ m x n ��Ԫ�صľ���m ��, n �У����밴��˳ʱ������˳�򣬷��ؾ����е�����Ԫ�ء�
    */
    // ��ʾ��ע��ת�Ǵ��������ϴα���
    vector<int> spiralOrder(vector<vector<int> >& matrix)
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
        ʵ�� pow(x, n) �������� x �� n ���ݺ�����
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
    // ��ĸ��λ�ʷ���
    /*
        ����һ���ַ������飬����ĸ��λ�������һ����ĸ��λ��ָ��ĸ��ͬ�������в�ͬ���ַ�����
    */
    vector<vector<string> > groupAnagrams(vector<string>& strs)
    {
        int len = strs.size();
        vector<vector<string> > ans;
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
    // ��תͼ��
    /*
        ����һ�� n �� n �Ķ�ά�����ʾһ��ͼ��
        ��ͼ��˳ʱ����ת 90 �ȡ�
        ˵����
        �������ԭ����תͼ������ζ������Ҫֱ���޸�����Ķ�ά�����벻Ҫʹ����һ����������תͼ��
    */
    // ע�⣺���ұ��Ǹ���������ת�����ڶ���ѭ����-1��
    // ����2�����Ӹ��Խ��߶Գƣ�Ȼ��ˮƽ�߶Գơ�
    // ԭ����Ϊ(x,y)����ת������Ϊ(y,n-x)���Գƺ�Ϊ(n-y,n-x)��ˮƽ�Գƺ�Ϊ(y,n-x)��
    void rotate(vector<vector<int> >& matrix)
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
    // ȫ����II
    /*
        ����һ���ɰ����ظ����ֵ����У��������в��ظ���ȫ���С�
    */
    // ��ʾ�����е���iλ���������iλ����Ȳ�ʹ�ù��ģ���ô��iλ�����ظ���
    vector<vector<int> > permuteUnique(vector<int>& nums)
    {
        sort(nums.begin(), nums.end());
        vector<vector<int> > ans;
        vector<int> result(nums.size());
        vector<bool> vis(nums.size());
        DFS_permuteUnique(ans, nums, vis, result, 0);
        return ans;
    }
    void DFS_permuteUnique(vector<vector<int> > &ans, vector<int> &nums, vector<bool> &vis, vector<int> &result, int step)
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
    // ȫ����
    /*
        ����һ��û���ظ����ֵ����У����������п��ܵ�ȫ���С�
    */
    bool v_permute[20];
    int result_permute[20];
    vector<vector<int> > permute(vector<int>& nums)
    {
        memset(v_permute, false, sizeof(v_permute));
        vector<vector<int> > ans;
        DFS_permute(nums, nums.size(), 0, ans);
        return ans;
    }
    void DFS_permute(vector<int>& nums, int length, int step, vector<vector<int> >& ans)
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
    // �ַ������
    /*
        �����������ַ�����ʽ��ʾ�ķǸ����� num1 �� num2������ num1 �� num2 �ĳ˻������ǵĳ˻�Ҳ��ʾΪ�ַ�����ʽ��
        ˵����
            num1 �� num2 �ĳ���С��110��
            num1 �� num2 ֻ�������� 0-9��
            num1 �� num2 �������㿪ͷ������������ 0 ����
            ����ʹ���κα�׼��Ĵ������ͣ����� BigInteger����ֱ�ӽ�����ת��Ϊ����������
    */
    // ע�⣺"123" * "0"
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
    // ����ܺ�II
    /*
        ����һ������ candidates ��һ��Ŀ���� target ���ҳ� candidates �����п���ʹ���ֺ�Ϊ target ����ϡ�
        candidates �е�ÿ��������ÿ�������ֻ��ʹ��һ�Ρ�
    */
    vector<vector<int> > combinationSum2(vector<int>& candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        vector<vector<int> > ans;
        vector<int> selections(target + 1);
        DFS_combinationSum2(ans, candidates, selections, target, 0, 0, 0);
        return ans;
    }
    void DFS_combinationSum2(vector<vector<int> > &ans, vector<int> &candidates, vector<int> &selections, int target, int sum, int step, int sta)
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
        for (int i = sta; i< candidates.size(); i++)
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
    // ����ܺ�
    /*
        ����һ�����ظ�Ԫ�ص����� candidates ��һ��Ŀ���� target ���ҳ� candidates �����п���ʹ���ֺ�Ϊ target ����ϡ�
        candidates �е����ֿ����������ظ���ѡȡ��
        ˵����
        �������֣����� target��������������
        �⼯���ܰ����ظ�����ϡ�
    */
    vector<vector<int> > combinationSum(vector<int>& candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        vector<vector<int> > ans;
        vector<int> selections(target + 1);
        DFS_combinationSum(ans, candidates, selections, target, 0, 0, 0);
        return ans;
    }
    void DFS_combinationSum(vector<vector<int> > &ans, vector<int> &candidates, vector<int> &selections, int target, int sum, int step, int sta)
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
        for (int i = sta; i< candidates.size(); i++)
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
    // ��Ч������
    /*
        �ж�һ�� 9x9 �������Ƿ���Ч��ֻ��Ҫ�������¹�����֤�Ѿ�����������Ƿ���Ч���ɡ�
        ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
        ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
        ���� 1-9 ��ÿһ���Դ�ʵ�߷ָ��� 3x3 ����ֻ�ܳ���һ�Ρ�
    */
    bool isValidSudoku(vector<vector<char> >& board)
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
    // �����������в���Ԫ�صĵ�һ�������һ��λ��
    /*
        ����һ�������������е��������� nums����һ��Ŀ��ֵ target���ҳ�����Ŀ��ֵ�������еĿ�ʼλ�úͽ���λ�á�
        ����㷨ʱ�临�Ӷȱ����� O(log n) ����
        ��������в�����Ŀ��ֵ������ [-1, -1]��
    */
    vector<int> searchRange(vector<int>& nums, int target)
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
    // ������ת��������
    /*
        ���谴�����������������Ԥ��δ֪��ĳ�����Ͻ�������ת��
        ( ���磬���� [0,1,2,4,5,6,7] ���ܱ�Ϊ [4,5,6,7,0,1,2] )��
        ����һ��������Ŀ��ֵ����������д������Ŀ��ֵ���򷵻��������������򷵻� -1 ��
        ����Լ��������в������ظ���Ԫ�ء�
        ����㷨ʱ�临�Ӷȱ����� O(log n) ����
    */
    // ��ʾ���������ظ�Ԫ�أ�ע����������������������Ұ���.
    // ����м����С�����ұߵ��������Ұ��������ģ����м����������ұ������������������
    int search(vector<int>& nums, int target)
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
    // ��һ������
    /*
        ʵ�ֻ�ȡ��һ�����еĺ������㷨��Ҫ���������������������г��ֵ�������һ����������С�
        �����������һ����������У��������������г���С�����У����������У���
        ����ԭ���޸ģ�ֻ����ʹ�ö��ⳣ���ռ䡣
    */
    void nextPermutation(vector<int>& nums)
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
    // �������
    /*
        �������������������� dividend �ͳ��� divisor�������������Ҫ��ʹ�ó˷��������� mod �������
        ���ر����� dividend ���Գ��� divisor �õ����̡�
    */
    // ��ʾ������1�����֣��˷���λ������㣻����2���ƽ���
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
    // �������������еĽڵ�
    /*
        ����һ���������������������ڵĽڵ㣬�����ؽ����������
        �㲻��ֻ�ǵ����ĸı�ڵ��ڲ���ֵ��������Ҫʵ�ʵĽ��нڵ㽻����
    */
    // ��ʾ��ע��[], [1], [1, 2, 3]
    ListNode* swapPairs(ListNode* head)
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
    // ��������
    /*
        ���� n �����������ŵĶ���������д��һ��������ʹ���ܹ��������п��ܵĲ�����Ч��������ϡ�
    */
    vector<string> generateParenthesis(int n)
    {
        vector<string> ans;
        DFS_generateParenthesis(ans, "", 0, 0, n);
        return ans;

    }
    void DFS_generateParenthesis(vector<string>& ans, string s, int left, int length, int n)
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
    // ɾ������ĵ�����N���ڵ�
    /*
        ����һ������ɾ������ĵ����� n ���ڵ㣬���ҷ��������ͷ��㡣
    */
    ListNode* removeNthFromEnd(ListNode* head, int n)
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
    // ����֮��
    /*
        ����һ������ n ������������ nums ��һ��Ŀ��ֵ target���ж� nums ���Ƿ�����ĸ�Ԫ�� a��b��c �� d ��ʹ�� a + b + c + d ��ֵ�� target ��ȣ��ҳ��������������Ҳ��ظ�����Ԫ�顣
    */
    // ��ʾ���̶���������˫ָ����������������������֮�ͣ���˫ָ��
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


    // 17
    // �绰�������ĸ���
    /*
        ����һ������������ 2-9 ���ַ����������������ܱ�ʾ����ĸ��ϡ�
        �������ֵ���ĸ��ӳ�����£���绰������ͬ����ע�� 1 ����Ӧ�κ���ĸ��
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
    // ��ӽ�������֮��
    /*
        ����һ������ n ������������ nums �� һ��Ŀ��ֵ target���ҳ� nums �е�����������ʹ�����ǵĺ��� target ��ӽ����������������ĺ͡��ٶ�ÿ������ֻ����Ψһ�𰸡�
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
    // ����֮��
    /*
        ����һ������ n ������������ nums���ж� nums ���Ƿ��������Ԫ�� a��b��c ��ʹ�� a + b + c = 0 ���ҳ��������������Ҳ��ظ�����Ԫ�顣
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


    // 12
    // ����ת��������
    /*
        �������ְ������������ַ��� I�� V�� X�� L��C��D �� M��
        �ַ�          ��ֵ
        I             1
        V             5
        X             10
        L             50
        C             100
        D             500
        M             1000
        ���磬 �������� 2 д�� II ����Ϊ�������е� 1��12 д�� XII ����Ϊ X + II �� 27 д��  XXVII, ��Ϊ XX + V + II ��
        ͨ������£�����������С�������ڴ�����ֵ��ұߡ���Ҳ�������������� 4 ��д�� IIII������ IV������ 1 ������ 5 ����ߣ�����ʾ�������ڴ��� 5 ��С�� 1 �õ�����ֵ 4 ��ͬ���أ����� 9 ��ʾΪ IX���������Ĺ���ֻ�������������������
        I ���Է��� V (5) �� X (10) ����ߣ�����ʾ 4 �� 9��
        X ���Է��� L (50) �� C (100) ����ߣ�����ʾ 40 �� 90��
        C ���Է��� D (500) �� M (1000) ����ߣ�����ʾ 400 �� 900��
        ����һ������������תΪ�������֡�����ȷ���� 1 �� 3999 �ķ�Χ�ڡ�
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
    // ʢ���ˮ������
    /*
        ���� n ���Ǹ����� a1��a2��...��an��ÿ�������������е�һ���� (i, ai) ���������ڻ� n ����ֱ�ߣ���ֱ�� i �������˵�ֱ�Ϊ (i, ai) �� (i, 0)���ҳ����е������ߣ�ʹ�������� x �Ṳͬ���ɵ�����������������ˮ��
        ˵�����㲻����б�������� n ��ֵ����Ϊ 2��
    */
    // ��ʾ��˫ָ�뷨������һ���ƶ�
    int maxArea(vector<int>& height)
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
    // �ַ���ת����
    /*
        ������Ѱ�ҵ��ĵ�һ���ǿ��ַ�Ϊ�����߸���ʱ���򽫸÷�����֮���澡���ܶ���������������������Ϊ�������������ţ������һ���ǿ��ַ������֣���ֱ�ӽ�����֮�������������ַ�����������γ�������
        ���ַ���������Ч����������֮��Ҳ���ܻ���ڶ�����ַ�����Щ�ַ����Ա����ԣ����Ƕ��ں�����Ӧ�����Ӱ�졣
        ע�⣺������ַ����еĵ�һ���ǿո��ַ�����һ����Ч�����ַ����ַ���Ϊ�ջ��ַ����������հ��ַ�ʱ������ĺ�������Ҫ����ת����
        ���κ�����£����������ܽ�����Ч��ת��ʱ���뷵�� 0��
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
    // Z���α任
    /*
        ��һ�������ַ������ݸ������������Դ������¡������ҽ��� Z �������С�
        ���������ַ���Ϊ "LEETCODEISHIRING" ����Ϊ 3 ʱ���������£�
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
    // ������Ӵ�
    /*
        ����һ���ַ��� s���ҵ� s ����Ļ����Ӵ�������Լ��� s ����󳤶�Ϊ 1000��
    */
    string longestPalindrome(string s)
    {
        // O��n���㷨
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


        // �����㷨
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
    // ���ظ��ַ�����Ӵ�
    /*
        ����һ���ַ����������ҳ����в������ظ��ַ��� ��Ӵ� �ĳ��ȡ�
    */
    // ��ʾ��ע���ж�(c[s[i]] != -1 && c[s[i]] >= start)
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
    // �������
    /*
        �������� �ǿ� ������������ʾ�����Ǹ������������У����Ǹ��Ե�λ���ǰ��� ���� �ķ�ʽ�洢�ģ��������ǵ�ÿ���ڵ�ֻ�ܴ洢 һλ ���֡�
        ��������ǽ��������������������᷵��һ���µ���������ʾ���ǵĺ͡�
        �����Լ���������� 0 ֮�⣬���������������� 0 ��ͷ��
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
};
int main()
{
    Solution* solution = new Solution();
    return 0;
}
