package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Activator;

import java.util.Random;

/**
 * 层
 * @author ZhangZhenfang
 * @date 2019/2/27 22:12
 */
abstract class Layer {

    /**
     * 上一层
     */
    protected Layer pre;
    /**
     * 下一层
     */
    protected Layer next;
    /**
     * 输入
     */
    protected Mat input;
    /**
     * 权重
     */
    protected Mat weight;
    /**
     * 线性输出
     */
    protected Mat z;
    /**
     * 经过复合函数后的输出
     */
    protected Mat a;
    /**
     * 偏置
     */
    protected Mat bia;
    /**
     * 激活函数
     */
    protected Activator activator;
    /**
     * 误差
     */
    protected Mat delta;
    /**
     * 梯度
     */
    protected Mat grad;


    public Layer() {}
    public Layer(int inputSize, int outSize, Activator activator) {
        this.activator = activator;
        double[] data = new double[inputSize * outSize];
        for (int i = 0; i < inputSize * outSize; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        this.weight = new Mat(inputSize, outSize, CvType.CV_32F);
        weight.put(0, 0, data);
        this.bia = new Mat(1, outSize, CvType.CV_32F);
        data = new double[outSize];
        for (int i = 0; i < outSize; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        bia.put(0, 0, data);
        this.z = new Mat();
        this.a = new Mat(1, outSize, CvType.CV_32F);
        this.delta = new Mat();
        this.grad = new Mat();
    }

    /**
     * 计算输出
     */
    public abstract void computeOut();

    /**
     * 计算梯度
     */
    public abstract void computeGrad();

    /**
     * 更新权重
     */
    public abstract void updateWeight(double rate);
    public Layer getPre() {
        return pre;
    }

    public void setPre(Layer pre) {
        this.pre = pre;
    }

    public Layer getNext() {
        return next;
    }

    public void setNext(Layer next) {
        this.next = next;
    }

    public Mat getInput() {
        return input;
    }

    public void setInput(Mat input) {
        this.input = input;
    }

    public Mat getWeight() {
        return weight;
    }

    public void setWeight(Mat weight) {
        this.weight = weight;
    }

    public Mat getZ() {
        return z;
    }

    public void setZ(Mat z) {
        this.z = z;
    }

    public Mat getA() {
        return a;
    }

    public void setA(Mat a) {
        this.a = a;
    }

    public Mat getBia() {
        return bia;
    }

    public void setBia(Mat bia) {
        this.bia = bia;
    }

    public Activator getActivator() {
        return activator;
    }

    public void setActivator(Activator activator) {
        this.activator = activator;
    }

    public Mat getDelta() {
        return delta;
    }

    public void setDelta(Mat delta) {
        this.delta = delta;
    }

    public Mat getGrad() {
        return grad;
    }

    public void setGrad(Mat grad) {
        this.grad = grad;
    }
}
