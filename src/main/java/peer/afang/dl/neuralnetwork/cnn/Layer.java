package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Activator;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/2/27 22:12
 */
public class Layer {

    protected Layer pre;
    protected Layer next;
    protected Mat input;
    protected Mat weight;
    protected Mat z;
    protected Mat a;
    protected double bia;
    protected Activator activator;
    protected Mat delta;
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
        this.bia = 0;
    }


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

    public double getBia() {
        return bia;
    }

    public void setBia(double bia) {
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
