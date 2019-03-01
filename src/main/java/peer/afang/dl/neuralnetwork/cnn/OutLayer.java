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

    @Override
    public void computeOut() {
        Core.gemm(pre.getA(), weight, 1, new Mat(), 1, z);
        Core.add(z, bia, z);
        activator.activate(z, a);
    }

    @Override
    public void computeGrad(){};

    public void computeGrad(Mat label) {
        Mat derivative = activator.derivative(a);
        Mat m = new Mat();
        Core.subtract(label, a, m);
        Core.multiply(m, derivative, delta);
        Core.gemm(pre.getA().t(), delta, 1, new Mat(), 1, grad);
    }
    /**
     * 更新weight
     */
    @Override
    public void updateWeight(double rate) {
        Mat m = new Mat();
        Core.multiply(grad, new Scalar(rate), m);
        Core.add(weight, m, weight);
        Core.multiply(delta, new Scalar(rate), m);
        Core.add(bia, m, bia);
    }
}
