using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public interface IPerceptron {
        double Threshold { get; set; }
        double[] Weights { get; }
        double this[int Index] { get; set; }

        int Length { get; }
        Func<double, double> Activation { get; set; }
        Func<double, double> DiffActivation { get; set; }

        double Compute(params double[] Input);
        double Compute(List<double> Input);
    }
}
