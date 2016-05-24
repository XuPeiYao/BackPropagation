using BackPropagation.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class HyperPerceptron : IPerceptron {
        public double this[int Index] {
            get {
                return Weights[Index];
            }
            set {
                Weights[Index] = value;
            }
        }

        public Func<double, double> Activation { get; set; }
            = x => x;
            //= ActivationFunction.LogisticSigmoid;

        public Func<double, double> DiffActivation { get; set; }
            = x => 1;

        public PerceptronNetwork Network { get; set; }

        public double Threshold { get; set; }

        public double[] Weights { get; set; }

        public int Length => Weights.Length;

        public HyperPerceptron(PerceptronNetwork Network) {
            if (Network == null) throw new ArgumentException();
            this.Network = Network;
            this.Weights = Enumerable.Range(0, Network.InputCount).Select(x=>1.0).ToArray();
        }

        public HyperPerceptron(PerceptronNetwork Network, double RandomMinWeight, double RandomMaxWeight) : this(Network) {
            Random rand = new Random(DateTime.Now.Millisecond);
            this.Weights = this.Weights.Select(x => rand.NextDouble(RandomMinWeight, RandomMaxWeight)).ToArray();
        }

        public double Compute(params double[] Input) {
            var networkResult = Network.Compute(Input);
            return Activation(networkResult.Select((x,i)=>x*Weights[i]).Sum() - Threshold);
        }

        public double Compute(List<double> Input) => Compute(Input.ToArray());
    }
}
