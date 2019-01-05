using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static DaggerOffer.Node;

namespace DaggerOffer
{
    class Solutions
    {
        /*
         * 快排
         */

        /*
         * 树节点包含左儿子、右儿子、父节点，求中序遍历的下一个节点
         * 如果为空，返回空
         * 如果有右孩子，返回右孩子的最左孩子
         * 否则，往上遍历，如果该节点是左孩子，那么下个就是父节点。否则继续往上遍历。
         */
        public TreeLinkNode GetNext(TreeLinkNode pNode)
        {
            if (pNode == null)
            {
                return pNode;
            }
            if (pNode.right != null)
            {
                TreeLinkNode node = pNode.right;
                while (node.left != null)
                {
                    node = node.left;
                }
                return node;
            }
            else
            {
                while (pNode.next != null)
                {
                    TreeLinkNode fa = pNode.next;
                    if (fa.left == pNode)
                    {
                        return fa;
                    }
                    else
                    {
                        pNode = fa;
                    }
                }
            }
            return null;
        }

        /*
         * 不用加减乘除做加法
         * ^相当于两数相加不进位，&（乘以2）相当于两数相加的进位。重复进行两个的结果和。
         */
        public int Add(int num1, int num2)
        {
            while (num2 != 0)
            {
                int sum1 = num1 ^ num2;
                int sum2 = (num1 & num2) << 1;
                num1 = sum1;
                num2 = sum2;
            }
            return num1 == 0 ? num2 : num1;
        }

        /*
         * 不用乘除、循环、if、swtich等，求1-n的和
         * 利用短路运算思想、幂、短路运算计算乘法
         */
        public int Sum_Solution(int n)
        {
            int ans = n;
            bool sum = ans != 0 && (ans = ans + Sum_Solution(n - 1)) > 0;
            return ans;
        }
        //public int Sum_Solution(int n)
        //{
        //    return Do_Sum_Solution(n, n + 1) >> 1;
        //}
        //private int Do_Sum_Solution(int a, int b)
        //{
        //    int ans = 0;
        //    bool sum1 = (a & 1) == 1 && (ans = ans + b) > 0;
        //    a >>= 1;
        //    b <<= 1;
        //    bool sum2 = (a != 0) && (ans = ans + Do_Sum_Solution(a, b)) > 0;
        //    return ans;
        //}

        /*
         * 判断是否是顺子
         */
        public bool IsContinuous(int[] numbers)
        {
            if (numbers == null)
            {
                return false;
            }
            int len = numbers.Length;
            if (len == 0)
            {
                return false;
            }

            Array.Sort(numbers);
            int mi = numbers[len - 1];
            int ma = numbers[0];
            bool ans = true;
            for (int i = 0; i < len; i++)
            {
                if (numbers[i] == 0)
                {
                    continue;
                }
                if (numbers[i] < mi)
                {
                    mi = numbers[i];
                }
                if (numbers[i] > ma)
                {
                    ma = numbers[i];
                }
                if (i > 0 && numbers[i] == numbers[i - 1])
                {
                    ans = false;
                    break;
                }
            }
            return (ans && ma - mi - 1 <= len - 2);
        }

        /*
         * 二叉搜索树转化成排序双向链表
         */
        private TreeNode last, head;
        public TreeNode Convert(TreeNode pRootOfTree)
        {
            last = null;
            head = null;
            Do_Convert(pRootOfTree);
            return head;
        }
        private void Do_Convert(TreeNode node)
        {
            if (node == null)
            {
                return;
            }
            Do_Convert(node.left);
            if (last == null)
            {
                last = node;
                head = node;
            }
            else
            {
                last.right = node;
                node.left = last;
                last = node;
            }
            Do_Convert(node.right);
        }

        /*
         * 复杂链表的复制
         */
        public RandomListNode Clone(RandomListNode pHead)
        {
            if (pHead == null)
            {
                return null;
            }
            RandomListNode node, newNode, next, newHead = null;
            node = pHead;
            while (node != null)
            {
                newNode = new RandomListNode(node.label);
                if (newHead == null)
                {
                    newHead = newNode;
                }
                next = node.next;
                node.next = newNode;
                newNode.next = next;
                node = next;
            }
            node = pHead;
            while (node != null)
            {
                newNode = node.next;
                newNode.random = node.random == null ? null : node.random.next;
                node = newNode.next;
            }
            node = pHead;
            while (node != null)
            {
                newNode = node.next;
                node.next = newNode.next;
                node = newNode.next;
                newNode.next = newNode.next == null ? null : newNode.next.next;
            }
            return newHead;
        }

        /*
         * 给出先序遍历和中序遍历，重构二叉树
         */
        public TreeNode reConstructBinaryTree(int[] pre, int[] tin)
        {
            if (pre == null || tin == null)
            {
                return null;
            }
            return Do_reConstructBinaryTree(pre, tin, 0, pre.Length - 1, 0, tin.Length - 1);
        }
        private TreeNode Do_reConstructBinaryTree(int[] pre, int[] tin, int preS, int preE, int tinS, int tinE)
        {
            if (preS > preE || tinS > tinE)
            {
                return null;
            }
            TreeNode node = new TreeNode(pre[preS]);
            for (int i = tinS; i <= tinE; i++)
            {
                if (tin[i] == pre[preS])
                {
                    node.left = Do_reConstructBinaryTree(pre, tin, preS + 1, preS + i - tinS, tinS, i - 1);
                    node.right = Do_reConstructBinaryTree(pre, tin, preS + i - tinS + 1, preE, i + 1, tinE);
                    break;
                }
            }
            return node;
        }


        /*
         * 从根节点到叶结点的和为sum的路径
         */
        private List<List<int>> results;
        private List<int> result;
        public List<List<int>> FindPath(TreeNode root, int expectNumber)
        {
            results = new List<List<int>>();
            result = new List<int>();
            Do_FindPath(root, 0, expectNumber);
            return results;
        }
        private void Do_FindPath(TreeNode node, int sum, int expectNumber)
        {
            if (sum > expectNumber)
            {
                return;
            }
            if (node == null)
            {
                return;
            }
            result.Add(node.val);
            if (node.left == null && node.right == null)
            {
                if (sum + node.val == expectNumber)
                {
                    results.Add(new List<int>(result));
                }
                result.RemoveAt(result.Count - 1);
                return;
            }
            Do_FindPath(node.left, sum + node.val, expectNumber);
            Do_FindPath(node.right, sum + node.val, expectNumber);
            result.RemoveAt(result.Count - 1);
        }

        /*
         * 判断二叉树是否为镜像二叉树
         */
        public bool isSymmetrical(TreeNode pRoot)
        {
            if (pRoot == null)
            {
                return true;
            }
            return Do_isSymmetrical(pRoot.left, pRoot.right);
        }
        private bool Do_isSymmetrical(TreeNode left, TreeNode right)
        {
            if (left == null && right == null)
            {
                return true;
            }
            if (left == null || right == null)
            {
                return false;
            }
            if (left.val != right.val)
            {
                return false;
            }
            return Do_isSymmetrical(left.left, right.right) && Do_isSymmetrical(left.right, right.left);
        }

        /*
         * 给出一个数组，每个数大小在0-（len-1），求一个重复的数
         */
        public bool duplicate(int[] numbers, int[] duplication)
        {
            if (numbers == null)
            {
                return false;
            }
            int len = numbers.Length;
            for (int i = 0; i < len; i++)
            {
                int num = numbers[i];
                if (num >= len)
                {
                    num -= len;
                }
                if (numbers[num] >= len)
                {
                    duplication[0] = num;
                    return true;
                }
                numbers[num] = numbers[num] + len;
            }
            return false;
        }

        /*
         * 字符串转化为整数
         */
        public int StrToInt(string str)
        {
            if (string.IsNullOrEmpty(str))
            {
                return 0;
            }
            int addOperation = -1;
            int subOperation = -1;
            bool correct = true;
            int ans = 0;
            for (int i = 0; i < str.Length; i++)
            {
                char c = str[i];
                if (c == '+')
                {
                    addOperation = i;
                }
                else if (c == '-')
                {
                    subOperation = i;
                }
                else if (!(c >= '0' && c <= '9'))
                {
                    correct = false;
                    break;
                }
                else
                {
                    ans = ans * 10 + (int)(c - '0');
                }
            }
            if (!correct)
            {
                return 0;
            }
            if (addOperation != -1 && addOperation != 0)
            {
                return 0;
            }
            if (subOperation != -1 && subOperation != 0)
            {
                return 0;
            }
            if (subOperation == 0)
            {
                return ans * -1;
            }
            return ans;
        }

        /*
         * 连续和为sum
         */
        public List<List<int>> FindContinuousSequence(int sum)
        {
            List<List<int>> results = new List<List<int>>();
            if (sum < 1)
            {
                return results;
            }
            for (int i = 1; i < sum; i++)
            {
                int L = i, R = sum;
                while (L <= R)
                {
                    int mid = (L + R) >> 1;
                    int total = (i + mid) * (mid - i + 1) / 2;
                    if (total == sum)
                    {
                        List<int> result = new List<int>();
                        for (int j = i; j <= mid; j++)
                        {
                            result.Add(j);
                        }
                        results.Add(result);
                        break;
                    }
                    else if (total > sum)
                    {
                        R = mid - 1;
                    }
                    else
                    {
                        L = mid + 1;
                    }
                }
            }
            return results;
        }

        /*
         * 判断序列是否为搜索二叉树的后序遍历
         */
        public bool VerifySquenceOfBST(int[] sequence)
        {
            if (sequence == null)
            {
                return false;
            }
            int length = sequence.Length;
            if (length == 0)
            {
                return false;
            }
            return Do_VerifySquenceOfBST(sequence, 0, length - 1);
        }
        private bool Do_VerifySquenceOfBST(int[] sequence, int L, int R)
        {
            if (L >= R)
            {
                return true;
            }
            int index = sequence[R];
            int leftR = L - 1;
            for (int i = L; i < R; i++)
            {
                if (sequence[i] < index)
                {
                    leftR = i;
                }
                else
                {
                    break;
                }
            }
            for (int i = leftR + 1; i < R; i++)
            {
                if (sequence[i] < index)
                {
                    return false;
                }
            }
            return Do_VerifySquenceOfBST(sequence, L, leftR) && Do_VerifySquenceOfBST(sequence, leftR + 1, R - 1);
        }


        /*
         * 给出一个入栈顺序，判断另外一个顺序是否是出栈顺序
         */
        public bool IsPopOrder(int[] pushV, int[] popV)
        {
            if (pushV == null)
            {
                return true;
            }

            int length = pushV.Length;
            int[] stack = new int[length];
            int top = -1, pushed = 0, index = 0;
            while (pushed < length)
            {
                while (top >= 0 && index < length && stack[top] == popV[index])
                {
                    top--;
                    index++;
                }
                stack[++top] = pushV[pushed++];
            }
            while (top >= 0 && index < length && stack[top] == popV[index])
            {
                top--;
                index++;
            }
            return index == length;
        }

        /*
         * 旋转数组中的最小值
         */
        public int minNumberInRotateArray(int[] rotateArray)
        {
            if (rotateArray == null)
            {
                return 0;
            }
            int length = rotateArray.Length;
            if (length == 0)
            {
                return 0;
            }

            if (rotateArray[0] < rotateArray[length - 1])
            {
                return rotateArray[0];
            }
            else
            {
                int L = 0, R = length - 1, mid, ans = rotateArray[0];
                while (L <= R)
                {
                    mid = (L + R) >> 1;
                    if (rotateArray[0] <= rotateArray[mid])
                    {
                        L = mid + 1;
                    }
                    else
                    {
                        ans = rotateArray[mid];
                        R = mid - 1;
                    }
                }
                return ans;
            }
        }

        /*
         * 有两个数出现一次，其他数都出现两次
         */
        public void FindNumsAppearOnce(int[] array, int[] num1, int[] num2)
        {
            if (array == null)
            {
                return;
            }
            int length = array.Length;
            if (length == 0)
            {
                return;
            }

            int sum = 0;
            for (int i = 0; i < length; i++)
            {
                sum ^= array[i];
            }

            int k = 1;
            while (true)
            {
                if ((sum & 1) == 1)
                {
                    break;
                }
                sum >>= 1;
                k <<= 1;
            }

            int sum1 = 0, sum2 = 0;
            for (int i = 0; i < length; i++)
            {
                if ((array[i] & k) == k)
                {
                    sum1 ^= array[i];
                }
                else
                {
                    sum2 ^= array[i];
                }
            }

            num1[0] = sum1;
            num2[0] = sum2;
        }

        /*
         * 顺时针打印矩阵
         */
        public List<int> printMatrix(int[][] matrix)
        {
            List<int> result = new List<int>();
            if (matrix == null)
            {
                return result;
            }

            int lengthRow = matrix.Length;
            if (lengthRow == 0)
            {
                return result;
            }

            int lengthCol = matrix[0].Length;
            int cur = 0;
            while (true)
            {
                int topRow = cur;
                int bottomRow = lengthRow - cur - 1;
                int leftCol = cur;
                int rightCol = lengthCol - cur - 1;
                if (topRow > bottomRow || leftCol > rightCol)
                {
                    break;
                }

                for (int i = cur; i < lengthCol - cur; i++)
                {
                    result.Add(matrix[cur][i]);
                }

                for (int i = cur + 1; i < lengthRow - cur; i++)
                {
                    result.Add(matrix[i][lengthCol - 1 - cur]);
                }

                if (bottomRow > topRow)
                {
                    for (int i = lengthCol - 2 - cur; i >= cur; i--)
                    {
                        result.Add(matrix[lengthRow - 1 - cur][i]);
                    }
                }

                if (rightCol > leftCol)
                {
                    for (int i = lengthRow - 2 - cur; i > cur; i--)
                    {
                        result.Add(matrix[i][cur]);
                    }
                }

                cur++;
            }
            return result;
        }

        /*
         * 判断平衡二叉树
         */
        private bool isBalanced;
        public bool IsBalanced_Solution(TreeNode pRoot)
        {
            isBalanced = true;
            Depth(pRoot);
            return isBalanced;
        }
        private int Depth(TreeNode node)
        {
            if (!isBalanced)
            {
                return 0;
            }
            if (node == null)
            {
                return 0;
            }
            int left = 1 + Depth(node.left);
            int right = 1 + Depth(node.right);
            if (left - right > 1 || right - left > 1)
            {
                isBalanced = false;
                return 0;
            }
            return left > right ? left : right;
        }

        /*
         * 翻转链表
         */
        public ListNode ReverseList(ListNode pHead)
        {
            ListNode head = null;
            while (pHead != null)
            {
                ListNode next = pHead.next;
                pHead.next = head;
                head = pHead;
                pHead = next;
            }
            return head;
        }

        /*
         * 调整数组顺序使奇数位于偶数前面
         */
        public int[] reOrderArray(int[] array)
        {
            if (array == null)
            {
                return array;
            }
            int len = array.Length;
            int[] ans = new int[len];
            int k = 0;
            foreach (int val in array)
            {
                if (val % 2 == 1)
                {
                    ans[k++] = val;
                }
            }
            foreach (int val in array)
            {
                if (val % 2 == 0)
                {
                    ans[k++] = val;
                }
            }
            return ans;
        }

        /*
         * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。 
         */
        private Stack<int> s1 = new Stack<int>();
        private Stack<int> s2 = new Stack<int>();
        public void push(int node)
        {
            s1.Push(node);
        }

        public int pop()
        {
            while (s1.Count != 0)
            {
                s2.Push(s1.Pop());
            }
            int ans = s2.Pop();
            while (s2.Count != 0)
            {
                s1.Push(s2.Pop());
            }
            return ans;
        }

        /*
         * 按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
         */
        public List<List<int>> Print2(TreeNode pRoot)
        {
            List<List<int>> result = new List<List<int>>();
            if (pRoot == null)
            {
                return result;
            }
            List<int> temp = new List<int>();
            int depth = 1;
            Queue<TreeNodeWithDepth> queue = new Queue<TreeNodeWithDepth>();
            queue.Enqueue(new TreeNodeWithDepth(pRoot, 1));
            while (queue.Count != 0)
            {
                TreeNodeWithDepth cur = queue.Dequeue();
                if (cur.depth == depth)
                {
                    temp.Add(cur.node.val);
                }
                else
                {
                    depth++;
                    List<int> row = new List<int>();
                    if ((depth & 1) == 1)
                    {
                        for (int i = temp.Count - 1; i >= 0; i--)
                        {
                            row.Add(temp[i]);
                        }
                    }
                    else
                    {
                        foreach (int val in temp)
                        {
                            row.Add(val);
                        }
                    }
                    result.Add(row);
                    temp.Clear();
                    temp.Add(cur.node.val);
                }
                if (cur.node.left != null)
                {
                    queue.Enqueue(new TreeNodeWithDepth(cur.node.left, cur.depth + 1));
                }
                if (cur.node.right != null)
                {
                    queue.Enqueue(new TreeNodeWithDepth(cur.node.right, cur.depth + 1));
                }
            }
            List<int> last = new List<int>();
            if ((depth & 1) != 1)
            {
                for (int i = temp.Count - 1; i >= 0; i--)
                {
                    last.Add(temp[i]);
                }
            }
            else
            {
                foreach (int val in temp)
                {
                    last.Add(val);
                }
            }
            result.Add(last);
            return result;
        }

        /*
         * 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
         */
        public List<List<int>> Print(TreeNode pRoot)
        {
            List<List<int>> result = new List<List<int>>();
            if (pRoot == null)
            {
                return result;
            }
            List<int> temp = new List<int>();
            int depth = 1;
            Queue<TreeNodeWithDepth> queue = new Queue<TreeNodeWithDepth>();
            queue.Enqueue(new TreeNodeWithDepth(pRoot, 1));
            while (queue.Count != 0)
            {
                TreeNodeWithDepth cur = queue.Dequeue();
                if (cur.depth == depth)
                {
                    temp.Add(cur.node.val);
                }
                else
                {
                    depth++;
                    List<int> row = new List<int>();
                    foreach(int val in temp)
                    {
                        row.Add(val);
                    }
                    result.Add(row);
                    temp.Clear();
                    temp.Add(cur.node.val);
                }
                if (cur.node.left != null)
                {
                    queue.Enqueue(new TreeNodeWithDepth(cur.node.left, cur.depth + 1));
                }
                if (cur.node.right != null)
                {
                    queue.Enqueue(new TreeNodeWithDepth(cur.node.right, cur.depth + 1));
                }
            }
            result.Add(temp);
            return result;
        }

        /*
         * 翻转单词。“I am” -> "am I"
         */
        public string ReverseSentence(string str)
        {
            if (string.IsNullOrEmpty(str))
            {
                return str;
            }
            int len = str.Length;
            char[] s = new char[len];
            for (int i = 0; i < len; i++)
            {
                s[i] = str[i];
            }

            int L = 0, R;
            while (L < len)
            {
                if (s[L] != ' ')
                {
                    R = L;
                    while(R < len && s[R] != ' ')
                    {
                        R++;
                    }
                    int l = L, r = R - 1;
                    while(l < r)
                    {
                        char temp = s[l];
                        s[l] = s[r];
                        s[r] = temp;
                        l++;
                        r--;
                    }
                    L = R;
                }
                else
                {
                    L++;
                }
            }
            L = 0;
            R = len - 1;
            while(L < R)
            {
                char temp = s[L];
                s[L] = s[R];
                s[R] = temp;
                L++;
                R--;
            }

            string result = "";
            for (int i = 0; i < len; i++)
            {
                result += s[i];
            }
            return result;
        }

        /*
         * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”
         */
        public string LeftRotateString(string str, int n)
        {
            if (string.IsNullOrEmpty(str))
            {
                return str;
            }
            int len = str.Length;
            n = (n % len + len) % len;
            if (n == 0)
            {
                return str;
            }

            char[] s = new char[len];
            for (int i = 0; i < len; i++)
            {
                s[i] = str[i];
            }

            int L = 0, R = len - 1;
            char temp;
            while (L < R)
            {
                temp = s[L];
                s[L] = s[R];
                s[R] = temp;
                L++;
                R--;
            }

            R = len - 1;
            L = R - n + 1;
            while (L < R)
            {
                temp = s[L];
                s[L] = s[R];
                s[R] = temp;
                L++;
                R--;
            }

            R = len - 1 - n;
            L = 0;
            while (L < R)
            {
                temp = s[L];
                s[L] = s[R];
                s[R] = temp;
                L++;
                R--;
            }

            string result = "";
            for (int i = 0; i < len; i++)
            {
                result += s[i];
            }
            return result;
        }

        /*
         * 给出一个增序数组，找到两个数的和为sum的，存在多组，返回乘积最小的 
         */
        public List<int> FindNumbersWithSum(int[] array, int sum)
        {
            List<int> result = new List<int>();
            if (array == null)
            {
                return result;
            }
            int L = 0, R = array.Length - 1;
            while (L < R)
            {
                if (array[L] + array[R] > sum)
                {
                    R--;
                }
                else if (array[L] + array[R] < sum)
                {
                    L++;
                }
                else
                {
                    if (result.Count == 0 || (long)array[L] * array[R] > (long)result[0] * result[1])
                    {
                        if (result.Count == 0)
                        {
                            result.Add(array[L]);
                            result.Add(array[R]);
                        }
                        else
                        {
                            result[0] = array[L];
                            result[1] = array[R];
                        }
                    }
                    L++;
                    R--;
                }
            }
            return result;
        }

        /*
         * 二叉树的深度
         */
        public int TreeDepth(TreeNode pRoot)
        {
            return BinaryTreeDepth(pRoot);
        }

        /*
         * 统计数字在排序数组中出现的次数
         */
        public int GetNumberOfK(int[] data, int k)
        {
            if (data == null)
            {
                return 0;
            }
            int len = data.Length;
            int first = -1, last = -1;
            int L = 0, R = len - 1, mid;
            while (L <= R)
            {
                mid = (L + R) >> 1;
                if (data[mid] == k)
                {
                    first = mid;
                    R = mid - 1;
                }
                else if (data[mid] > k)
                {
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }
            L = 0;
            R = len - 1;
            while (L <= R)
            {
                mid = (L + R) >> 1;
                if (data[mid] == k)
                {
                    last = mid;
                    L = mid + 1;
                }
                else if (data[mid] > k)
                {
                    R = mid - 1;
                }
                else
                {
                    L = mid + 1;
                }
            }

            if (first == -1)
            {
                return 0;
            }
            return last - first + 1;
        }

        /*
         *  两个链表的第一个公共节点 
         */
        public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2)
        {
            if (pHead1 == null || pHead2 == null)
            {
                return null;
            }
            int len1 = 0, len2 = 0, len;
            ListNode head1 = pHead1;
            ListNode head2 = pHead2;
            while(head1 != null)
            {
                len1++;
                head1 = head1.next;
            }
            while (head2 != null)
            {
                len2++;
                head2 = head2.next;
            }            
            if (len1 > len2)
            {
                head1 = pHead1;
                head2 = pHead2;
                len = len1 - len2;
            }
            else
            {
                head1 = pHead2;
                head2 = pHead1;
                len = len2 - len1;
            }

            while(len > 0)
            {
                head1 = head1.next;
                len--;
            }
            while (head1 != null)
            {
                if (head1 == head2)
                {
                    return head1;
                }
                head1 = head1.next;
                head2 = head2.next;
            }
            return null;
        }

        /*
         * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
         */
        public List<int> PrintFromTopToBottom(TreeNode root)
        {
            List<int> result = new List<int>();
            if (root != null)
            {
                Queue<TreeNode> queue = new Queue<TreeNode>();
                queue.Enqueue(root);
                while (queue.Count != 0)
                {
                    TreeNode node = queue.Dequeue();
                    result.Add(node.val);
                    if (node.left != null)
                    {
                        queue.Enqueue(node.left);
                    }
                    if (node.right != null)
                    {
                        queue.Enqueue(node.right);
                    }
                }
            }
            return result;
        }

        /*
         * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
         */
        public string PrintMinNumber(int[] numbers)
        {
            int a = 1;
            List<int> f = new List<int>();
            for (int i = 0; i < 9; i++)
            {
                f.Add(a);
                a = a * 10;
            }
            Array.Sort(numbers, new Comparare());
            string ans = "";
            foreach (int x in numbers)
            {
                //Console.WriteLine(x);
                ans += x.ToString();
            }
            return ans;
        }

        /*
         * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
         */
        public double Power(double thebase, int exponent)
        {
            int newExponent = exponent > 0 ? exponent : -exponent;
            double ans = 1.0;
            while (newExponent != 0)
            {
                if ((newExponent & 1) == 1)
                {
                    ans = ans * thebase;
                }
                thebase = thebase * thebase;
                exponent >>= 1;
            }
            if (exponent < 0)
            {
                ans = 1 / ans;
            }
            return ans;
        }

        /*
         * ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
         */
        public int NumberOf1Between1AndN_Solution(int n)
        {
            string nStr = n.ToString();
            int len = nStr.Length;
            List<int> f = new List<int>();
            int a = 1;
            for (int i = 0; i <= len; i++)
            {
                f.Add(a);
                a = a * 10;
            }
            int ans = 0;
            for (int i = 0; i < len; i++)
            {
                int num = len - i - 1;
                for (int j = 0; j + '0' < nStr[i]; j++)
                {
                    if (j == 1)
                    {
                        ans += f[num];
                    }
                    if (num > 0)
                    {
                        ans = ans + f[num - 1] * num;
                    }
                }
                if (nStr[i] == '1')
                {
                    ans = ans + n % f[num] + 1;
                }
            }
            return ans;
        }

        /*
         * 给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
         */
        public int FindGreatestSumOfSubArray(int[] array)
        {
            if (array.Length == 0)
            {
                return 0;
            }
            int sum = 0, ma = array[0];
            foreach (int x in array)
            {
                sum += x;
                if (sum > ma)
                {
                    ma = sum;
                }
                if (sum < 0)
                {
                    sum = 0;
                }
            }
            return ma;
        }

        /*
         * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
         */
        public int MoreThanHalfNum_Solution(int[] numbers)
        {
            int len = numbers.Length;
            int num = 0, index = -1;
            foreach (int x in numbers)
            {
                if (x != index)
                {
                    if (num == 0)
                    {
                        index = x;
                        num = 1;
                    }
                    else
                    {
                        num--;
                    }
                }
                else
                {
                    num++;
                }
            }
            num = 0;
            foreach (int x in numbers)
            {
                if (x == index)
                {
                    num++;
                }
            }
            if (num > len / 2)
            {
                return index;
            }
            else
            {
                return 0;
            }
        }

        /*
         * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
         */
        public List<string> ans;
        public bool[] v = new bool[10];
        public char[] s = new char[10];
        public HashSet<string> hash = new HashSet<string>();

        public void dfs(string cur, int len)
        {
            if (cur.Length == len)
            {
                if (hash.Contains(cur))
                {
                    return;
                }
                hash.Add(cur);
                ans.Add(cur);
                return;
            }
            for (int i = 0; i < len; i++)
            {
                if (!v[i])
                {
                    v[i] = true;
                    dfs(cur + s[i], len);
                    v[i] = false;
                }
            }
        }

        public List<string> Permutation(string str)
        {
            ans = new List<string>();
            int len = str.Length;
            if (len == 0)
            {
                return ans;
            }
            char c;
            for (int i = 0; i < len; i++)
            {
                s[i] = str[i];
                v[i] = false;
            }
            for (int i = 0; i < len; i++)
            {
                for (int j = i + 1; j < len; j++)
                {
                    if (s[i] > s[j])
                    {
                        c = s[j];
                        s[j] = s[i];
                        s[i] = c;
                    }
                }
            }

            dfs("", len);
            return ans;
        }

        /*
         * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
         */
        public ListNode Merge(ListNode pHead1, ListNode pHead2)
        {
            ListNode head = null;
            ListNode cur = null;
            while (pHead1 != null && pHead2 != null)
            {
                if (pHead1.val < pHead2.val)
                {
                    if (cur == null)
                    {
                        cur = pHead1;
                        head = pHead1;
                        pHead1 = pHead1.next;
                    }
                    else
                    {
                        cur.next = pHead1;
                        cur = cur.next;
                        pHead1 = pHead1.next;
                    }
                }
                else
                {
                    if (cur == null)
                    {
                        cur = pHead2;
                        head = pHead2;
                        pHead2 = pHead2.next;
                    }
                    else
                    {
                        cur.next = pHead2;
                        cur = cur.next;
                        pHead2 = pHead2.next;
                    }
                }
            }
            if (pHead1 != null)
            {
                if (head == null)
                {
                    head = pHead1;
                }
                else
                {
                    cur.next = pHead1;
                }
            }
            else if (pHead2 != null)
            {
                if (head == null)
                {
                    head = pHead2;
                }
                else
                {
                    cur.next = pHead2;
                }
            }
            return head;
        }

        /*
         * 输入一个链表，输出该链表中倒数第k个结点。
         */
        public ListNode FindKthToTail(ListNode head, int k)
        {
            ListNode ans = null;
            ListNode head1 = head;
            int num = 0;
            while (head1 != null)
            {
                num++;
                if (num == k)
                {
                    ans = head;
                }
                else if (num > k && ans != null)
                {
                    ans = ans.next;
                }
                head1 = head1.next;
            }
            return ans;
        }

        /*
         * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
         */
        public int NumberOf1(int n)
        {
            int ans = 0;
            while (n != 0)
            {
                n = n & (n - 1);
                ans++;
            }
            return ans;
        }

        /*
         * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
         */
        public int rectCover(int number)
        {
            int a = 1, b = 2, c = 0;
            if (number == 1)
            {
                return 1;
            }
            if (number == 2)
            {
                return 2;
            }
            for (int i = 3; i <= number; i++)
            {
                c = a + b;
                a = b;
                b = c;
            }
            return c;
        }

        /*
         * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
         */
        public int jumpFloorII(int number)
        {
            return 1 << (number - 1);
        }

        /*
         * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
         */
        public int jumpFloor(int number)
        {
            if (number == 1)
            {
                return 1;
            }
            if (number == 2)
            {
                return 2;
            }
            int a = 1, b = 2, c = 0;
            for (int i = 3; i <= number; i++)
            {
                c = a + b;
                a = b;
                b = c;
            }
            return c;
        }

        /*
         * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39
         */
        public int Fibonacci(int n)
        {
            if (n == 0)
            {
                return 0;
            }
            int a = 1, b = 1, c = 1;
            for (int i = 3; i <= n; i++)
            {
                c = a + b;
                a = b;
                b = c;
            }
            return c;
        }


        /*
         * 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList
         */
        public List<int> printListFromTailToHead(ListNode listNode)
        {
            List<int> result = new List<int>();
            ListNode current = listNode;
            while (current != null)
            {
                result.Add(current.val);
                current = current.next;
            }
            int L = 0, R = result.Count - 1;
            while (L < R)
            {
                int t = result[L];
                result[L] = result[R];
                result[R] = t;
                L++;
                R--;
            }
            return result;
        }

        /*
         * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
         */
        public string replaceSpace(string str)
        {
            return str.Replace(" ", "%20");
        }

        /*
         * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
         */
        public bool Find(int target, int[][] array)
        {
            if (array == null)
            {
                return false;
            }
            int len1 = array.Length;
            if (len1 == 0)
            {
                return false;
            }
            int len2 = array[0].Length;
            if (len2 == 0)
            {
                return false;
            }
            for (int i = 0; i < len1; i++)
            {
                if (target >= array[i][0] && target <= array[i][len2 - 1])
                {
                    int L = 0, R = len2 - 1;
                    while (L <= R)
                    {
                        int mid = (L + R);
                        if (array[i][mid] > target)
                        {
                            R = mid - 1;
                        }
                        else if (array[i][mid] < target)
                        {
                            L = mid + 1;
                        }
                        else
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        private int BinaryTreeDepth(TreeNode node)
        {
            if (node == null)
            {
                return 0;
            }
            return Math.Max(BinaryTreeDepth(node.left) + 1, BinaryTreeDepth(node.right) + 1);
        }
    }

    public class Node
    {
        public int val;
        public IList<Node> children;

        public Node() { }
        public Node(int _val, IList<Node> _children)
        {
            val = _val;
            children = _children;
        }
    }

    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int x)
        {
            val = x;
        }
    }

    public class TreeLinkNode
    {
        public int val;
        public TreeLinkNode left;
        public TreeLinkNode right;
        public TreeLinkNode next;
        public TreeLinkNode(int x)
        {
            val = x;
        }
    }

    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x)
        {
            val = x;
        }
    }

    public class TreeNodeWithDepth
    {
        public TreeNode node;
        public int depth;
        public TreeNodeWithDepth(TreeNode node, int depth)
        {
            this.node = node;
            this.depth = depth;
        }
    }

    public class RandomListNode
    {
        public int label;
        public RandomListNode next, random;
        public RandomListNode(int x)
        {
            this.label = x;
        }
    }

    class Comparare : IComparer<int>
    {
        public int Compare(int x, int y)
        {
            string xStr = x.ToString();
            string yStr = y.ToString();
            return string.Compare(xStr + yStr, yStr + xStr);
        }
    }
}
