using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public interface IPerceptron {
        double Threshold { get; }
        double[] Weights { get; }
        double this[int Index] { get; }

        int Length { get; }
        Func<double, double> Activation { get; }
        Func<double, double> DiffActivation { get; }

        double Compute(params double[] Input);
    }
}
