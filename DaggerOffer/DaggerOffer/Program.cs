using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DaggerOffer
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] a = { 4, 1, 3, 2, 5, 0, 8, 7, 2, 9 };
            Common common = new Common();
            common.Bubble_Sort(a, 0, 9);
            foreach (int i in a)
            {
                Console.WriteLine(i);
            }
        }
    }

    public class Test1
    {
        public static int a;
        public int b;
        public Test1()
        {
            a = 2;
        }

        public void Cal()
        {
            a++;
        }

        public void Print()
        {
            Console.WriteLine(a);
        }
    }
}
