using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public static class ActivationFunction {
        public static double LogisticSigmoid(double x) {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double Diff_LogisticSigmoid(double x) {
            return Math.Exp(-x) * Math.Pow(1 + Math.Exp(-x), -2);
        }


        public static double HyperbolicTangent(double x) {
            return Math.Tanh(x);
        }

        public static double Diff_HyperbolicTangent(double x) {
            return Math.Pow(2 / (Math.Exp(x) + Math.Exp(-x)), 2);
        }


    }
}
