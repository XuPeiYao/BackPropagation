using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation.Extensions {
    public static class RandomExtension {
        public static double NextDouble(this Random rand, double Min, double Max) {
            return (rand.NextDouble() * (Max - Min)) + Min;
        }
    }
}
