using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DaggerOffer
{
    class Common
    {
        /*
         * 插入排序
         */
        public void Insert_Sort(int[] a, int start, int end)
        {
            if (start >= end)
            {
                return;
            }

        }

        /*
         * 冒泡排序
         */
        public void Bubble_Sort(int[] a, int start, int end)
        {
            if (start >= end)
            {
                return;
            }
            for (int i = 0; i < end - start + 1; i++)
            {
                for (int j = start; j < end - i; j++)
                {
                    if (a[j] >= a[j + 1])
                    {
                        int temp = a[j];
                        a[j] = a[j + 1];
                        a[j + 1] = temp;
                    }
                }
            }
        }

        /*
         * 归并排序
         */
        public void Merge_Sort(int[] a, int start, int end)
        {
            if (start >= end)
            {
                return;
            }
            int mid = (start + end) / 2;
            Merge_Sort(a, start, mid);
            Merge_Sort(a, mid + 1, end);
            int[] temp = new int[end - start + 1];
            int index = 0, left = start, right = mid + 1;
            while (left <= mid && right <= end)
            {
                if (a[left] <= a[right])
                {
                    temp[index++] = a[left++];
                }
                else
                {
                    temp[index++] = a[right++];
                }
            }
            while (left <= mid)
            {
                temp[index++] = a[left++];
            }
            while (right <= end)
            {
                temp[index++] = a[right++];
            }
            index = 0;
            for (int i = start; i <= end; i++)
            {
                a[i] = temp[index++];
            }
        }

        /*
         * 快速排序
         * 非递归写法
         * 优化：1. 每次选key可以选第一个、中间、最后一个的中间值；2. 长度较短，可以使用插入排序
         */
        public void Quick_Sort(int[] a, int start, int end)
        {
            if (start >= end)
            {
                return;
            }
            int index = Partial_Quick_Sort(a, start, end);
            Quick_Sort(a, start, index - 1);
            Quick_Sort(a, index + 1, end);
        }
        //public void Quick_Sort(int[] a, int start, int end)
        //{
        //    if (start >= end)
        //    {
        //        return;
        //    }
        //    Stack<int> stack = new Stack<int>();
        //    stack.Push(start);
        //    stack.Push(end);
        //    while (stack.Count != 0)
        //    {
        //        int newEnd = stack.Pop();
        //        int newStart = stack.Pop();
        //        if (newStart >= newEnd)
        //        {
        //            continue;
        //        }
        //        int index = Partial_Quick_Sort(a, newStart, newEnd);
        //        stack.Push(newStart);
        //        stack.Push(index - 1);
        //        stack.Push(index + 1);
        //        stack.Push(newEnd);
        //    }
        //}
        public int Partial_Quick_Sort(int[] a, int start, int end)
        {
            int key = a[end];
            while (start < end)
            {
                while (start < end && a[start] <= key)
                {
                    start++;
                }
                a[end] = a[start];
                while (start < end && a[end] >= key)
                {
                    end--;
                }
                a[start] = a[end];
            }
            a[end] = key;
            return end;
        }
    }
}
