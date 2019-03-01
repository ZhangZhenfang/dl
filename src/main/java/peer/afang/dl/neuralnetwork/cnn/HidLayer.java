package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.Activator;
import peer.afang.dl.util.MatUtils;

/**
 * @author ZhangZhenfang
 * @date 2019/2/27 21:02
 */
public class HidLayer extends Layer{

    public HidLayer(int inputSize, int outSize, Activator activator) {
        super(inputSize, outSize, activator);
    }

    /**
     * 计算输出
     */
    @Override
    public void computeOut() {
        Core.gemm(pre.getA(), weight, 1, new Mat(), 1, z);
        Core.add(z, bia, z);
        activator.activate(z, a);
    }

    /**
     * 计算梯度
     */
    @Override
    public void computeGrad() {
        Mat derivative = activator.derivative(a);
        Mat m = new Mat();
        Core.gemm(next.getWeight(), next.getDelta().t(), 1, new Mat(), 1, m);
        Core.multiply(m.t(), derivative, delta);
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
