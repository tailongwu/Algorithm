using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static DaggerOffer.Node;

namespace DaggerOffer
{
    class LeetCode
    {
        // Problem500
        public string[] FindWords(string[] words)
        {
            if (words == null)
            {
                return words;
            }
            string s1 = "qwertyuiop";
            string s2 = "asdfghjkl";
            string s3 = "zxcvbnm";
            Dictionary<char, int> dic = new Dictionary<char, int>();
            foreach (char c in s1)
            {
                dic.Add(c, 1);
            }
            foreach (char c in s2)
            {
                dic.Add(c, 2);
            }
            foreach (char c in s3)
            {
                dic.Add(c, 4);
            }
            List<string> list = new List<string>();
            foreach (string word in words)
            {
                int row = 0;
                foreach (char c in word)
                {
                    if (c >= 'A' && c <= 'Z')
                    {
                        row |= dic[(char)(c + ('z' - 'Z'))];
                    }
                    else
                    {
                        row |= dic[c];
                    }
                }
                if (row == 1 || row == 2 || row == 4)
                {
                    list.Add(word);
                }
            }
            return list.ToArray();
        }

        // Problem811
        public IList<string> SubdomainVisits(string[] cpdomains)
        {
            List<string> results = new List<string>();
            if (cpdomains == null)
            {
                return results;
            }
            Dictionary<string, int> dic = new Dictionary<string, int>();
            foreach (string domain in cpdomains)
            {
                string[] parts = domain.Split(' ');
                int num = int.Parse(parts[0]);
                string[] ds = parts[1].Split('.');
                int len = ds.Length;
                string d = "";
                for (int i = len - 1; i >= 0; i--)
                {
                    d = ds[i] + d;
                    if (dic.ContainsKey(d))
                    {
                        dic[d] = dic[d] + num;
                    }
                    else
                    {
                        dic.Add(d, num);
                    }
                    d = '.' + d;
                }
            }
            foreach (var i in dic)
            {
                results.Add(string.Format("{0} {1}", i.Value, i.Key));
            }
            return results;
        }

        // Problem559
        public int MaxDepth(Node root)
        {
            if (root == null)
            {
                return 0;
            }
            int ma = 1;
            foreach (Node node in root.children)
            {
                int dep = MaxDepth(node) + 1;
                if (dep > ma)
                {
                    ma = dep;
                }
            }
            return ma;
        }

        // Problem557
        public string ReverseWords(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }
            int len = s.Length;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < len;)
            {
                if (s[i] != ' ')
                {
                    int j = i;
                    while (j < len && s[j] != ' ')
                    {
                        j++;
                    }
                    for (int k = j - 1; k >= i; k--)
                    {
                        sb.Append(s[k]);
                    }
                    i = j;
                }
                else
                {
                    sb.Append(' ');
                    i++;
                }
            }
            return sb.ToString();
        }

        // Problem700
        public TreeNode SearchBST(TreeNode root, int val)
        {
            if (root == null)
            {
                return null;
            }
            if (root.val == val)
            {
                return root;
            }
            if (val > root.val)
            {
                return SearchBST(root.right, val);
            }
            return SearchBST(root.left, val);
        }

        // Problem589
        public IList<int> Preorder(Node root, IList<int> ans = null)
        {
            if (ans == null)
            {
                ans = new List<int>();
            }
            if (root == null)
            {
                return ans;
            }
            ans.Add(root.val);
            foreach (Node node in root.children)
            {
                Preorder(node, ans);
            }
            return ans;
        }

        // Problem590
        public IList<int> Postorder(Node root, IList<int> ans = null)
        {
            if (ans == null)
            {
                ans = new List<int>();
            }
            if (root == null)
            {
                return ans;
            }
            foreach (Node node in root.children)
            {
                Postorder(node, ans);
            }
            ans.Add(root.val);
            return ans;
        }

        // Problem876
        public ListNode MiddleNode(ListNode head)
        {
            if (head == null)
            {
                return head;
            }
            ListNode node1 = head;
            ListNode node2 = head.next;
            while (node2 != null)
            {
                node1 = node1.next;
                node2 = node2.next;
                if (node2 != null)
                {
                    node2 = node2.next;
                }
            }
            return node1;
        }

        // Problem908
        public int SmallestRangeI(int[] A, int K)
        {
            if (A == null || A.Length == 0)
            {
                return 0;
            }
            int mi = A[0];
            int ma = A[0];
            foreach (int a in A)
            {
                if (a > ma)
                {
                    ma = a;
                }
                else if (a < mi)
                {
                    mi = a;
                }
            }
            if (ma - mi < 2 * K)
            {
                return 0;
            }
            return ma - mi - 2 * K;
        }

        // Problem922: 重排数组，下标为奇数放奇数，下标为偶数放偶数
        public int[] SortArrayByParityII(int[] A)
        {
            if (A == null)
            {
                return A;
            }
            int len = A.Length;
            int[] B = new int[len];
            int i = 0, j = 1;
            foreach (int a in A)
            {
                if ((a & 1) == 0)
                {
                    B[i] = a;
                    i += 2;
                }
                else
                {
                    B[j] = a;
                    j += 2;
                }
            }
            return B;
        }

        // Problem832: 每一行翻转并且1变0,0变1
        public int[][] FlipAndInvertImage(int[][] A)
        {
            if (A == null || A.Length == 0 || A[0] == null || A[0].Length == 0)
            {
                return A;
            }
            int cols = A[0].Length;
            foreach (int[] a in A)
            {
                int L = 0;
                int R = cols - 1;
                while (L < R)
                {
                    a[L] = a[L] ^ a[R];
                    a[R] = a[L] ^ a[R];
                    a[L] = a[L] ^ a[R];
                    a[L] ^= 1;
                    a[R] ^= 1;
                    L++;
                    R--;
                }
                if (L == R)
                {
                    a[L] ^= 1;
                }
            }
            return A;
        }

        // Problem905: 给定一个数组，然后重排，将偶数排在前，奇数在后
        public int[] SortArrayByParity(int[] A)
        {
            if (A == null)
            {
                return A;
            }
            int R = A.Length - 1;
            int L = 0;
            while (L < R)
            {
                if (A[L] % 2 == 1 && A[R] % 2 == 0)
                {
                    A[L] = A[L] ^ A[R];
                    A[R] = A[L] ^ A[R];
                    A[L] = A[L] ^ A[R];
                    L++;
                    R--;
                }
                else if (A[L] % 2 == 0)
                {
                    L++;
                }
                else
                {
                    R--;
                }
            }
            return A;
        }
    }
}
