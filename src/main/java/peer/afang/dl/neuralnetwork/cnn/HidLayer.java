package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
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
    public void computeOut() {
        z = new Mat();
        Core.gemm(pre.getA(), weight, 1, new Mat(), 1, z);
//        Core.add(z, new Scalar(bia), z);
        a = activator.activate(z);
    }

    /**
     * 计算梯度
     */
    public void computeGrad() {
        Mat derivative = activator.derivative(a);
        delta = new Mat();
        grad = new Mat();
        Core.gemm(next.getDelta(), next.getWeight().t(), 1, new Mat(), 1, delta);
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
