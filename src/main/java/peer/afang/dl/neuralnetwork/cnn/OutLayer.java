package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.Activator;
import peer.afang.dl.util.MatUtils;

/**
 * @author ZhangZhenfang
 * @date 2019/2/27 21:44
 */
public class OutLayer extends Layer{

    public OutLayer(int inputSize, int outSize, Activator activator) {
        super(inputSize, outSize, activator);
    }

    public void computeOut() {
        z = new Mat();
        Core.gemm(pre.getA(), weight, 1, new Mat(), 1, z);
        a = activator.activate(z);
    }

    public void computeGrad(Mat label) {
//        System.out.println(z.dump());
        Mat derivative = activator.derivative(a);
        delta = new Mat();
        grad = new Mat();
        Core.subtract(label, a, delta);
        Core.multiply(delta, derivative, delta);
        Core.gemm(pre.a.t(), delta, 1, new Mat(), 1, grad);
    }
    /**
     * 更新weight
     */
    public void updateWeight(double rate) {
        Core.multiply(grad, new Scalar(rate), grad);
        Core.add(weight, grad, weight);
//        bia -= rate * MatUtils.sumMat(delta);
    }
}
