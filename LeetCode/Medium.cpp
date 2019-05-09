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
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
 };
class Solution
{
public:
    // 797
    // ���п���·��
    /*
        ��һ���� n �����������޻�ͼ���ҵ����д� 0 �� n-1 ��·�����������Ҫ��˳��
        ��ά����ĵ� i �������еĵ�Ԫ����ʾ����ͼ�� i �Ž�����ܵ������һЩ��㣨����ע������ͼ���з���ģ����涨��a��b��Ͳ��ܴ�b��a���վ���û����һ������ˡ�
    */
    vector<vector<int> > allPathsSourceTarget(vector<vector<int> >& graph)
    {
        vector<vector<int> > ans;
        vector<int> result;
        result.push_back(0);
        DFS_allPathsSourceTarget(graph, ans, result, 0);
        return ans;
    }
    void DFS_allPathsSourceTarget(vector<vector<int> > &graph, vector<vector<int> > &ans, vector<int> &result, int k)
    {
        if (k == graph.size() - 1)
        {
            ans.push_back(result);
        }
        for (int i = 0; i < graph[k].size(); i++)
        {
            if (graph[k][i] > k)
            {
                result.push_back(graph[k][i]);
                DFS_allPathsSourceTarget(graph, ans, result, graph[k][i]);
                result.pop_back();
            }
        }
    }


    // 863
    // �����������о���Ϊ K �Ľ��
    /*
        ����һ�������������и���� root���� һ��Ŀ���� target ����һ������ֵ K ��
        ���ص�Ŀ���� target ����Ϊ K �����н���ֵ���б� �𰸿������κ�˳�򷵻ء�
    */
    // ��ʾ�����ҳ���Ŀ����path��Ȼ�����ÿ���ڵ����һ�����ӣ�ע�⵱ǰ�ڵ�Ҳ�����Ǵ�
    vector<int> distanceK(TreeNode* root, TreeNode* target, int K)
    {
        vector<int> ans;
        vector<TreeNode*> path;
        FindPath_distanceK(root, target, path);
        int len = path.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *node;
            int D;
            if (i != len - 1)
            {
                if (path[i + 1] != path[i]->left)
                {
                    node = path[i]->left;
                }
                else
                {
                    node = path[i]->right;
                }
                D = K - (len - i);
            }
            else
            {
                node = path[i];
                D = K;
            }
            if (D >= 0)
            {
                FindAns_distanceK(node, ans, D, 0);
            }
            else if (D == -1)
            {
                ans.push_back(path[i]->val);
            }
        }
        return ans;
    }
    void FindPath_distanceK(TreeNode *node, TreeNode *target, vector<TreeNode*> &path)
    {
        if (node == 0 || (path.size() > 0 && path.back() == target))
        {
            return;
        }
        path.push_back(node);
        if (node == target)
        {
            return;
        }
        FindPath_distanceK(node->left, target, path);
        if (path.size() > 0 && path.back() == target)
        {
            return;
        }
        FindPath_distanceK(node->right, target, path);
        if (path.size() > 0 && path.back() == target)
        {
            return;
        }
        path.pop_back();
    }
    void FindAns_distanceK(TreeNode *node, vector<int> &ans, int K, int D)
    {
        if (node == 0)
        {
            return;
        }
        if (K == D)
        {
            ans.push_back(node->val);
            return;
        }
        FindAns_distanceK(node->left, ans, K, D + 1);
        FindAns_distanceK(node->right, ans, K, D + 1);
    }


    // 870
    // ����ϴ��
    /*
        ����������С��ȵ����� A �� B��A ����� B �����ƿ��������� A[i] > B[i] ������ i ����Ŀ��������
        ���� A ���������У�ʹ������� B ��������󻯡�
    */
    vector<int> advantageCount(vector<int>& A, vector<int>& B)
    {
        sort(A.begin(), A.end());
        int len = A.size();
        vector<int> ans(len);
        vector<bool> vis(len);
        for (int i = 0; i < len; i++)
        {
            vis[i] = false;
        }
        for (int i = 0; i < len; i++)
        {
            int L = 0, R = len - 1, mid, ansIndex = 0;
            while (L <= R)
            {
                mid = (L + R) >> 1;
                if (A[mid] > B[i])
                {
                    ansIndex = mid;
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }
            while (true)
            {
                if (!vis[ansIndex])
                {
                    ans[i] = A[ansIndex];
                    vis[ansIndex] = true;
                    break;
                }
                ansIndex++;
                if (ansIndex == len)
                {
                    ansIndex = 0;
                }
            }
        }
        return ans;
    }


    // 306
    // �ۼ���
    /*
        �ۼ�����һ���ַ���������������ֿ����γ��ۼ����С�
        һ����Ч���ۼ����б������ٰ��� 3 �����������ʼ�����������⣬�ַ����е���������������֮ǰ��������ӵĺ͡�
        ����һ��ֻ�������� '0'-'9' ���ַ�������дһ���㷨���жϸ��������Ƿ����ۼ�����
        ˵��: �ۼ���������������� 0 ��ͷ�����Բ������ 1, 2, 03 ���� 1, 02, 3 �������
        ���磺112358��199100199
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


    // 445
    // �������II
    /*
        ���������ǿ����������������Ǹ��������������λλ������ʼλ�á����ǵ�ÿ���ڵ�ֻ�洢�������֡�����������ӻ᷵��һ���µ�����
        ����: ��������������޸ĸ���δ������仰˵���㲻�ܶ��б��еĽڵ���з�ת��
    */
    ListNode* addTwoNumbersII(ListNode* l1, ListNode* l2)
    {
        l1 = Reverse_addTwoNumbers(l1);
        l2 = Reverse_addTwoNumbers(l2);
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
        l1 = Reverse_addTwoNumbers(l1);
        l2 = Reverse_addTwoNumbers(l2);
        head = Reverse_addTwoNumbers(head);
        return head;
    }
    ListNode* Reverse_addTwoNumbers(ListNode *l)
    {
        ListNode *head = NULL, *cur = l, *next;
        while (cur != NULL)
        {
            next = cur->next;
            cur->next = head;
            head = cur;
            cur = next;
        }
        return head;
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


    // 454
    // �������II
    /*
        �����ĸ����������������б� A , B , C , D ,�����ж��ٸ�Ԫ�� (i, j, k, l) ��ʹ�� A[i] + B[j] + C[k] + D[l] = 0��
        Ϊ��ʹ����򵥻������е� A, B, C, D ������ͬ�ĳ��� N���� 0 �� N �� 500 �����������ķ�Χ�� -228 �� 228 - 1 ֮�䣬���ս�����ᳬ�� 231 - 1 ��
    */
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D)
    {
        int len = A.size();
        map<int, int> m;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                m[A[i] + B[j]]++;
            }
        }
        int ans = 0;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                int sum = -(C[i] + D[j]);
                if (m[sum] > 0)
                {
                    ans += m[sum];
                }
            }
        }
        return ans;
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


    // 923
    // ����֮�͵Ķ��ֿ���
    /*
        ����һ���������� A���Լ�һ������ target ��ΪĿ��ֵ���������� i < j < k �� A[i] + A[j] + A[k] == target ��Ԫ�� i, j, k ��������
        ���ڽ����ǳ����뷵�� ������� 10^9 + 7 ��������
    */
    // ��ʾ����������ͬ����������ͬ����������ͬ
    int threeSumMulti(vector<int>& A, int target)
    {
        long long ans = 0;
        int len = A.size();
        sort(A.begin(), A.end());
        vector<int> nums(101);
        for (int i = 0; i < 101; i++)
        {
            nums[i] = 0;
        }
        for (int i = 0; i < len; i++)
        {
            nums[A[i]]++;
        }
        int cur = 0;
        for (int i = 0; i < len; )
        {
            int j = i + 1;
            while (j < len && A[j] == A[i])
            {
                j++;
            }
            A[cur++] = A[i];
            i = j;
        }
        for (int i = 0; i < cur; i++)
        {
            int L = i + 1, R = cur - 1;
            while (L < R)
            {
                int sum = A[i] + A[L] + A[R];
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
                    ans = (ans + (long long)nums[A[i]] * nums[A[L]] * nums[A[R]]) % 1000000007;
                    L++;
                    R--;
                }
            }
        }
        for (int i = 0; i < 101; i++)
        {
            if (nums[i] >= 2)
            {
                int k = target - i * 2;
                if (k >= 0 && k < 101 && k != i && nums[k] > 0)
                {
                    ans = (ans + (long long)nums[i] * (nums[i] - 1) / 2 * nums[k]) % 1000000007;
                }
            }
        }
        if (target % 3 == 0)
        {
            int k = target / 3;
            if (k >= 0 && k < 101 && nums[k] >= 3)
            {
                ans = (ans + (long long)nums[k] * (nums[k] - 1) * (nums[k] - 2) / 6) % 1000000007;
            }
        }
        return (int)ans;
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

    // 650
    // ֻ���������ļ���
    /*
        �����һ�����±���ֻ��һ���ַ� 'A'����ÿ�ο��Զ�������±��������ֲ�����
        Copy All (����ȫ��) : ����Ը���������±��е������ַ�(���ֵĸ����ǲ������)��
        Paste (ճ��) : �����ճ������һ�θ��Ƶ��ַ���
        ����һ������ n ������Ҫʹ�����ٵĲ����������ڼ��±��д�ӡ��ǡ�� n �� 'A'������ܹ���ӡ�� n �� 'A' �����ٲ���������
    */
    int minSteps(int n)
    {
        // ��������30������ֽ��1*30=1+30��2*15=2+15��3*10=3+10��5*6=5+6��2*3*5=2+3+5��
        int ans = 0;
        if (n < 2)
        {
            return 0;
        }
        for (int i = 2; i <= n; i++)
        {
            if (n % i == 0)
            {
                while (n % i == 0)
                {
                    ans += i;
                    n /= i;
                }
            }
        }
        return ans;
    }
//    int minSteps(int n)
//    {
//        vector<int> dp(n + 1);
//        for (int i = 0; i <= n; i++)
//        {
//            dp[i] = 1e9;
//        }
//        dp[1] = 0;
//        for (int i = 2; i <= n; i++)
//        {
//            for (int j = 2; j <= i; j++)
//            {
//                if (i % j == 0)
//                {
//                    dp[i] = min(dp[i], dp[i / j] + i / j);
//                }
//            }
//        }
//        return dp[n];
//    }
//    struct Q_minSteps
//    {
//        int step, len, copyLen;
//    };
//    int minSteps(int n)
//    {
//        queue<Q_minSteps> Q;
//        Q_minSteps p, q;
//        p.step = 0;
//        p.len = 1;
//        p.copyLen = 0;
//        Q.push(p);
//        while (!Q.empty())
//        {
//            p = Q.front();
//            Q.pop();
//            if (p.len == n)
//            {
//                while (!Q.empty())
//                {
//                    Q.pop();
//                }
//                return p.step;
//            }
//            if (p.copyLen != 0)
//            {
//                q.len = p.len + p.copyLen;
//                q.step = p.step + 1;
//                q.copyLen = p.copyLen;
//                if (q.len <= n)
//                {
//                    Q.push(q);
//                }
//            }
//            if (p.len != p.copyLen)
//            {
//                q.len = p.len;
//                q.step = p.step + 1;
//                q.copyLen = p.len;
//                Q.push(q);
//            }
//        }
//        return 0;
//    }


    // 456
    // 132ģʽ
    /*
        ����һ���������У�a1, a2, ..., an��һ��132ģʽ�������� ai, aj, ak ������Ϊ���� i < j < k ʱ��ai < ak < aj�����һ���㷨���������� n �����ֵ�����ʱ����֤����������Ƿ���132ģʽ�������С�
    */
    bool find132pattern(vector<int>& nums)
    {
        int len = nums.size();
        if (len == 0)
        {
            return false;
        }

        vector<int> mi(len);
        mi[0] = nums[0];
        for (int i = 1; i < len; i++)
        {
            mi[i] = mi[i - 1] < nums[i] ? mi[i - 1] : nums[i];
        }
        stack<int> sta;
        int ma = 1e9;
        for (int i = len - 1; i > 0; i--)
        {
            if (nums[i] > mi[i])
            {
                while (!sta.empty() && sta.top() < nums[i])
                {
                    ma = sta.top();
                    sta.pop();
                }
                if (ma < nums[i] && ma > mi[i] < ma)
                {
                    return true;
                }
            }
            sta.push(nums[i]);
        }
        return false;
    }


    // 542
    // 01����
    /*
        ����һ���� 0 �� 1 ��ɵľ����ҳ�ÿ��Ԫ�ص������ 0 �ľ��롣
        ��������Ԫ�ؼ�ľ���Ϊ 1
    */
    vector<vector<int> > updateMatrix(vector<vector<int> >& matrix)
    {
        int row = matrix.size();
        int col = matrix[0].size();
        vector<vector<int> > ans(row);
        for (int i = 0; i < row; i++)
        {
            ans[i].resize(col);
        }
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (matrix[i][j] == 0)
                {
                    ans[i][j] = 0;
                    continue;
                }
                bool getAns = false;
                for (int k = 1; !getAns; k++)
                {
                    for (int x = 0; x <= k && !getAns; x++)
                    {
                        int y = k - x;
                        int xx = i + x;
                        int yy = j + y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i - x;
                        yy = j + y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i - x;
                        yy = j - y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                        xx = i + x;
                        yy = j - y;
                        if (xx >= 0 && yy >= 0 && xx < row && yy < col && matrix[xx][yy] == 0)
                        {
                            ans[i][j] = k;
                            getAns = true;
                            break;
                        }
                    }
                }
            }
        }
        return ans;
    }


    // 890
    // ���Һ��滻ģʽ
    /*
        ����һ�������б� words ��һ��ģʽ  pattern������֪�� words �е���Щ������ģʽƥ�䡣
        ���������ĸ������ p ��ʹ�ý�ģʽ�е�ÿ����ĸ x �滻Ϊ p(x) ֮�����Ǿ͵õ�������ĵ��ʣ���ô������ģʽ��ƥ��ġ�
        ������һ�£���ĸ�������Ǵ���ĸ����ĸ��˫�䣺ÿ����ĸӳ�䵽��һ����ĸ��û��������ĸӳ�䵽ͬһ����ĸ����
        ���� words �������ģʽƥ��ĵ����б�
        ����԰��κ�˳�򷵻ش𰸡�
    */
    vector<string> findAndReplacePattern(vector<string>& words, string pattern)
    {
        vector<string> ans;
        int si = words.size();
        int len = pattern.size();
        for (int i = 0; i < si; i++)
        {
            int len1 = words[i].size();
            if (len1 != len)
            {
                continue;
            }
            map<char, char> m;
            for (int j = 0; j < len1; j++)
            {
                m[words[i][j]] = 0;
            }
            bool yes = true;
            for (int j = 0; j < len1; j++)
            {
                if (!m[words[i][j]])
                {
                    m[words[i][j]] = pattern[j];
                }
                else if (m[words[i][j]] != pattern[j])
                {
                    yes = false;
                    break;
                }
            }
            if (yes)
            {
                ans.push_back(words[i]);
            }
        }
        return ans;
    }


    // 216
    // ����ܺ�III
    /*
        �ҳ��������֮��Ϊ n �� k ��������ϡ������ֻ������ 1 - 9 ��������������ÿ������в������ظ������֡�
    */
    bool v_combinationSum3[10];
    vector<int> a_combinationSum3;
    vector<vector<int> > combinationSum3(int k, int n)
    {
        memset(v_combinationSum3, false, sizeof(v_combinationSum3));
        a_combinationSum3.resize(10);
        vector<vector<int> > ans;
        do_combinationSum3(ans, n, k, 0, 0);
        return ans;
    }
    void do_combinationSum3(vector<vector<int> > &ans, int n, int k, int stp, int sum)
    {
        if (sum > n || k > 9)
        {
            return;
        }
        if (k == stp)
        {
            if (sum == n)
            {
                vector<int> result(k);
                for(int i = 0; i < k; i++)
                {
                    result[i] = a_combinationSum3[i];
                }
                ans.push_back(result);
            }
            return;
        }
        for (int i = stp == 0 ? 0 : a_combinationSum3[stp - 1]; i < 9; i++)
        {
            if (!v_combinationSum3[i])
            {
                v_combinationSum3[i] = true;
                a_combinationSum3[stp] = i + 1;
                do_combinationSum3(ans, n, k, stp + 1, sum + i + 1);
                v_combinationSum3[i] = false;
            }
        }
    }

    // 77
    // ���
    /*
        ������������ n �� k������ 1 ... n �����п��ܵ� k ��������ϡ�
    */
    vector<bool> v_combine;
    vector<int> a_combine;
    vector<vector<int> > combine(int n, int k)
    {
        v_combine.resize(n);
        a_combine.resize(k);
        for (int i = 0; i < n; i++)
        {
            v_combine[i] = false;
        }
        vector<vector<int> > ans;
        do_combine(ans, n, k, 0);
        return ans;
    }
    void do_combine(vector<vector<int> > &ans, int n, int k, int stp)
    {
        if (k == stp)
        {
            vector<int> result(k);
            for (int i = 0; i < k; i++)
            {
                result[i] = a_combine[i];
            }
            ans.push_back(result);
            return;
        }
        for (int i = stp == 0 ? 0 : a_combine[stp - 1]; i < n; i++)
        {
            if (!v_combine[i])
            {
                v_combine[i] = true;
                a_combine[stp] = i + 1;
                do_combine(ans, n, k, stp + 1);
                v_combine[i] = false;
            }
        }
    }

    // 894
    // ���п��ܵ���������
    /*
        ����������һ�������������ÿ�����ǡ���� 0 �� 2 ���ӽ�㡣
        ���ذ��� N ���������п��������������б� �𰸵�ÿ��Ԫ�ض���һ���������ĸ���㡣
        ����ÿ������ÿ����㶼������ node.val=0��
        ����԰��κ�˳�򷵻����������б�
    */
    vector<TreeNode*> allPossibleFBT(int N)
    {
        vector<TreeNode*> ans;
        if (N & 1)
        {
            do_allPossibleFBT(N, ans, 0, 0, 0);
        }
        return ans;
    }
    void do_allPossibleFBT(int N, vector<TreeNode*> ans, int step, TreeNode *node, TreeNode *root)
    {
        if (N < step)
        {
            return;
        }
        if (N == step)
        {
            ans.push_back(root);
            return;
        }
        if (node == 0)
        {
            node = new TreeNode(0);
            node->left = 0;
            node->right = 0;
            do_allPossibleFBT(N, ans, step + 1, node, node);
            return;
        }

        TreeNode *left = new TreeNode(0);
        left->left = 0;
        left->right = 0;
        TreeNode *right = new TreeNode(0);
        right->left = 0;
        right->right = 0;
        node->left = left;
        node->right = right;

        do_allPossibleFBT(N, ans, step + 2, left, root);
        do_allPossibleFBT(N, ans, step + 2, right, root);

        node->left = 0;
        node->right = 0;
    }


    // 46
    // ȫ����
    bool v_permute[20];
    int result_permute[20];
    vector<vector<int> > permute(vector<int>& nums)
    {
        memset(v_permute, false, sizeof(v_permute));
        vector<vector<int> > ans;
        do_permute(nums, nums.size(), 0, ans);
        return ans;
    }
    void do_permute(vector<int>& nums, int length, int step, vector<vector<int> >& ans)
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
                do_permute(nums, length, step + 1, ans);
                v_permute[i] = false;
            }
        }
    }


    // 814
    // ��������֦
    /*
        ��������������� root ����������ÿ������ֵҪô�� 0��Ҫô�� 1��
        �����Ƴ������в����� 1 ��������ԭ��������
    */
    TreeNode* pruneTree(TreeNode* root)
    {
        do_pruneTree(root);
        return root;
    }
    bool do_pruneTree(TreeNode* node)
    {
        if (node == 0)
        {
            return false;
        }
        bool left = do_pruneTree(node->left);
        bool right = do_pruneTree(node->right);
        if (!left)
        {
            node->left = 0;
        }
        if (!right)
        {
            node->right = 0;
        }
        if (node->val == 1 || left || right)
        {
            return true;
        }
        return false;
    }


    // 861
    // ��ת�����ĵ÷�
    /*
        ��һ����ά���� A ����ÿ��Ԫ�ص�ֵΪ 0 �� 1 ��
        �ƶ���ָѡ����һ�л��У���ת�����л����е�ÿһ��ֵ�������� 0 ������Ϊ 1�������� 1 ������Ϊ 0��
        ����������������ƶ��󣬽��þ����ÿһ�ж����ն������������ͣ�����ĵ÷־�����Щ���ֵ��ܺ͡�
        ���ؾ����ܸߵķ�����
    */
    int matrixScore(vector<vector<int> >& A)
    {
        int f[25];
        f[0] = 1;
        for (int i = 1; i < 25; i++)
        {
            f[i] = f[i - 1] << 1;
        }
        int row = A.size();
        int col = A[0].size();
        for (int i = 0; i < row; i++)
        {
            if (A[i][0] == 0)
            {
                for (int j = 0; j < col; j++)
                {
                    A[i][j] = 1 - A[i][j];
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < col; i++)
        {
            int cnt = 0;
            for (int j = 0; j < row; j++)
            {
                cnt += A[j][i];
            }
            if (cnt < row - cnt)
            {
                cnt = row - cnt;
            }
            ans = ans + cnt * f[col - 1 - i];
        }
        return ans;
    }


    // 866
    // ��������
    /*
        ������ڻ���� N ����С����������
    */
    // �Ż�������11��ż��λ�Ļ��Ĵ����ܱ�11����
    bool a_primePalindrome[20000];
    int p_primePalindrome[20000], total_primePalindrome;
    int f[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    int primePalindrome(int N)
    {
        if (N == 1)
        {
            return 2;
        }

        total_primePalindrome = 0;
        memset(a_primePalindrome, false, sizeof(a_primePalindrome));
        for (int i = 2; i < 20000; i++)
        {
            if (!a_primePalindrome[i])
            {
                p_primePalindrome[total_primePalindrome++] = i;
                for (int j = i * i; j < 20000; j += i)
                {
                    a_primePalindrome[j] = true;
                }
            }
        }

        if (N <= 100)
        {
            for (int i = N; ; i++)
            {
                if (do_primePalindrome1(i) && do_primePalindrome2(i))
                {
                    return i;
                }
            }
        }
        if ((N & 1) == 0)
        {
            N++;
        }
        int last = 0;
        for (int i = 0; i < 9; i++)
        {
            if (f[i] > N)
            {
                break;
            }
            last = i;
        }
        for (int i = N; ; i += 2)
        {
            if (f[last] < i)
            {
                last++;
                if (!(last & 1))
                {
                    i = f[last] - 1;
                    last++;
                    continue;
                }
            }
            if (do_primePalindrome2(i) && do_primePalindrome1(i))
            {
                return i;
            }
        }
    }
    bool do_primePalindrome1(int n)
    {
        int a[10], cnt=0;
        while (n != 0)
        {
            a[cnt++] = n % 10;
            n /= 10;
        }
        int L = 0, R = cnt -1;
        while (L < R)
        {
            if (a[L] != a[R])
            {
                return false;
            }
            L++;
            R--;
        }
        return true;
    }
    bool do_primePalindrome2(int n)
    {
        for (int i = 0; i < total_primePalindrome; i++)
        {
            if (p_primePalindrome[i] >= n)
            {
                return true;
            }
            if (n % p_primePalindrome[i] == 0)
            {
                return false;
            }
        }
        return true;
    }

    // 22
    // ��������
    /*
        ���� n �����������ŵĶ���������д��һ��������ʹ���ܹ��������п��ܵĲ�����Ч��������ϡ�
    */
    vector<string> generateParenthesis(int n)
    {
        vector<string> ans;
        do_generateParenthesis(ans, "", 0, 0, n);
        return ans;

    }
    void do_generateParenthesis(vector<string>& ans, string s, int left, int length, int n)
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
        do_generateParenthesis(ans, s + "(", left + 1, length + 1, n);
        do_generateParenthesis(ans, s + ")", left - 1, length + 1, n);
    }

    // 1008
    // ��������������������
    /*
        ���������������� preorder ��ƥ��Ķ�����������binary search tree���ĸ���㡣
    */
    TreeNode* bstFromPreorder(vector<int>& preorder)
    {
        return do_bstFromPreorder(preorder, 0, preorder.size() - 1);
    }
    TreeNode* do_bstFromPreorder(vector<int>& preorder, int L, int R)
    {
        if (L > R)
        {
            return 0;
        }
        if (L == R)
        {
            TreeNode* node = new TreeNode(preorder[L]);
            node->left = 0;
            node->right = 0;
            return node;
        }
        int index = L;
        for (int i = L + 1; i <= R; i++)
        {
            if (preorder[i] > preorder[L])
            {
                break;
            }
            index = i;
        }
        TreeNode* node = new TreeNode(preorder[L]);
        node->left = do_bstFromPreorder(preorder, L + 1, index);
        node->right = do_bstFromPreorder(preorder, index + 1, R);
        return node;
    }


    // 338
    // ����λ����
    /*
        ����һ���Ǹ����� num������ 0 �� i �� num ��Χ�е�ÿ������ i ����������������е� 1 ����Ŀ����������Ϊ���鷵�ء�
    */
    vector<int> countBits(int num)
    {
        vector<int> ans(num + 1);
        ans[0] = 0;
        for (int i = 1; i <= num; i++)
        {
            ans[i] = ans[i & (i - 1)] + 1;
//            if ((i & 1) == 0)
//            {
//                ans[i] = ans[i >> 1];
//            }
//            else
//            {
//                ans[i] = ans[i - 1] + 1;
//            }
        }
        return ans;
    }


    // 59
    // ��������2
    /*
        ����һ�������� n������һ������ 1 �� n2 ����Ԫ�أ���Ԫ�ذ�˳ʱ��˳���������е������ξ���
    */
    vector<vector<int> > generateMatrix(int n)
    {
        vector<vector<int> > ans(n);
        for (int i = 0; i < n; i++)
        {
            vector<int> result(n);
            for (int j = 0; j < n; j++)
            {
                result[j] = 0;
            }
            ans[i] = result;
        }
        int row = 0, col = 0, dir = 0;
        for (int i = 0; i < n * n; i++)
        {
            ans[row][col] = i + 1;
            if (dir == 0)
            {
                if (col + 1 >= n || ans[row][col + 1] != 0)
                {
                    dir = 1;
                    row++;
                }
                else
                {
                    col++;
                }
            }
            else if (dir == 1)
            {
                if (row + 1 >= n || ans[row + 1][col] != 0)
                {
                    dir = 2;
                    col--;
                }
                else
                {
                    row++;
                }
            }
            else if (dir == 2)
            {
                if (col - 1 < 0 || ans[row][col - 1] != 0)
                {
                    dir = 3;
                    row--;
                }
                else
                {
                    col--;
                }
            }
            else
            {
                if (row - 1 < 0 || ans[row - 1][col] != 0)
                {
                    dir = 0;
                    col++;
                }
                else
                {
                    row--;
                }
            }
        }
        return ans;
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

    // 654
    // ��������
    /*
        ����һ�������ظ�Ԫ�ص��������顣һ���Դ����鹹�������������������£�
        �������ĸ��������е����Ԫ�ء�
        ��������ͨ�����������ֵ��߲��ֹ����������������
        ��������ͨ�����������ֵ�ұ߲��ֹ����������������
        ͨ�����������鹹�������������������������ĸ��ڵ㡣
    */
    TreeNode* constructMaximumBinaryTree(vector<int>& nums)
    {
        return do_constructMaximumBinaryTree(nums, 0, nums.size() - 1);
    }
    TreeNode* do_constructMaximumBinaryTree(vector<int>& nums, int L, int R)
    {
        if (L > R)
        {
            return 0;
        }
        if (L == R)
        {
            TreeNode *node = new TreeNode(nums[L]);
            node->left = 0;
            node->right = 0;
            return node;
        }
        int index = L;
        for (int i = L; i <= R; i++)
        {
            if (nums[index] < nums[i])
            {
                index = i;
            }
        }
        TreeNode *node = new TreeNode(nums[index]);
        node->left = do_constructMaximumBinaryTree(nums, L, index - 1);
        node->right = do_constructMaximumBinaryTree(nums, index + 1, R);
        return node;
    }

    // 938
    // �����������ķ�Χ��
    /*
        ���������������ĸ���� root������ L �� R������֮������н���ֵ�ĺ͡�������������֤����Ψһ��ֵ
    */
    bool findL_rangeSumBST, findR_rangeSumBST;
    int sum_rangeSumBST;
    int rangeSumBST(TreeNode* root, int L, int R)
    {
        if (L == R)
        {
            return L;
        }

        findL_rangeSumBST = false;
        findR_rangeSumBST = false;
        sum_rangeSumBST = 0;
        do_rangeSumBST(root, L, R);
        return sum_rangeSumBST;
    }
    void do_rangeSumBST(TreeNode *root, int L, int R)
    {
        if (findL_rangeSumBST && findR_rangeSumBST)
        {
            return;
        }
        if (root == 0)
        {
            return;
        }
        do_rangeSumBST(root->left, L, R);
        if (root->val == L)
        {
            findL_rangeSumBST = true;
        }
        if (!findR_rangeSumBST && findL_rangeSumBST)
        {
            sum_rangeSumBST += root->val;
        }
        if (root->val == R)
        {
            findR_rangeSumBST = true;
        }
        do_rangeSumBST(root->right, L, R);
    }

    // 807
    // ���ֳ��������
    /*
        �ڶ�ά����grid�У�grid[i][j]����λ��ĳ���Ľ�����ĸ߶ȡ� ���Ǳ����������κ���������ͬ��������������ܲ�ͬ���Ľ�����ĸ߶ȡ� �߶� 0 Ҳ����Ϊ�ǽ����
        ��󣬴�������������ĸ����򣨼��������ײ��������Ҳࣩ�ۿ��ġ�����ߡ�������ԭʼ������������ͬ�� ���е�������Ǵ�Զ���ۿ�ʱ�������н������γɵľ��ε��ⲿ������ �뿴��������ӡ�
        ������߶ȿ������ӵ�����ܺ��Ƕ��٣�
    */
    int maxIncreaseKeepingSkyline(vector<vector<int> >& grid)
    {
        int row = grid.size();
        int col = grid[0].size();
        vector<int> ma(row);
        for (int i = 0; i < row; i++)
        {
            int x = grid[i][0];
            for (int j = 0; j < col; j++)
            {
                if (grid[i][j] > x)
                {
                    x = grid[i][j];
                }
            }
            ma[i] = x;
        }
        int ans = 0;
        for (int i = 0; i < col; i++)
        {
            int x = grid[0][i];
            for (int j = 0; j < row; j++)
            {
                if (x < grid[j][i])
                {
                    x = grid[j][i];
                }
            }
            for (int j = 0; j < row; j++)
            {
                int mi = x > ma[j] ? ma[j] : x;
                ans = ans + mi - grid[j][i];
            }
        }
        return ans;
    }

    // 950
    // ������˳����ʾ����
    /*
        �����е�ÿ�ſ��ƶ���Ӧ��һ��Ψһ������������԰�����Ҫ��˳������׿�Ƭ��������
        �������Щ�����������������泯�µģ�����δ��ʾ״̬����
        ���ڣ��ظ�ִ�����²��裬ֱ����ʾ���п���Ϊֹ��

        �����鶥����һ���ƣ���ʾ����Ȼ������������Ƴ���
        ��������������ƣ�����һ�Ŵ������鶥�����Ʒ�������ĵײ���
        �������δ��ʾ���ƣ���ô���ز��� 1������ֹͣ�ж���
        �������Ե���˳����ʾ���Ƶ�����˳��

        ���еĵ�һ���Ʊ���Ϊ�����ƶѶ�����
    */
    // ��ʾ������˼�����ҵ�����
    vector<int> deckRevealedIncreasing(vector<int>& deck)
    {
        int length = deck.size();
        vector<int> ans(length);
        sort(deck.begin(), deck.end());
        for (int i = length - 1; i >= 0; i--)
        {
            ans[i] = deck[i];
            int index = ans[length - 1];
            for (int j = length - 1; j > i + 1; j--)
            {
                ans[j] = ans[j - 1];
            }
            if (i != length - 1)
            {
                ans[i + 1] = index;
            }
        }
        return ans;
    }
};
int main()
{
    Solution* solution = new Solution();
    vector<int> a;
    a.push_back(2);
    a.push_back(7);
    a.push_back(11);
    a.push_back(15);
    vector<int> b;
    b.push_back(1);
    b.push_back(10);
    b.push_back(4);
    b.push_back(11);
    solution->advantageCount(a, b);
    return 0;
}
