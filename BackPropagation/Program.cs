using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    class Program {
        static void Main(string[] args) {
            PerceptronNetwork network = new PerceptronNetwork(2, 0, 1, new int[] {4,1 });
            LearnData[] data = new LearnData[] {
                new LearnData() { Input = new double[] {0,0 },Output = new double[] { 0 }},
                new LearnData() { Input = new double[] {1,1 },Output = new double[] { 0 }},
                new LearnData() { Input = new double[] {1,0 },Output = new double[] { 1 }},
                new LearnData() { Input = new double[] {0,1 },Output = new double[] { 1 }},
            };
            
            network.Train(data,0.8,0.05);
            Console.WriteLine(network.Compute(0, 0)[0]);
            Console.WriteLine(network.Compute(1, 1)[0]);

            Console.WriteLine(network.Compute(1, 0)[0]);
            Console.WriteLine(network.Compute(0, 1)[0]);

            StreamWriter writer = new StreamWriter("output.json");
            writer.Write(network.ToJObject());
            writer.Close();

            Console.ReadKey();
        }
    }
}
