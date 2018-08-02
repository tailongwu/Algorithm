using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DaggerOffer
{
    class Solutions
    {
        /*
         * 输入一个链表，反转链表后，输出新链表的表头。
         */
        public ListNode ReverseList(ListNode pHead)
        {
            ListNode newHead = pHead;
            while(pHead != null)
            {
                if (newHead == pHead)
                {

                }
            }
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
}
