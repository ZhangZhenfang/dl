package peer.afang.dl.perception;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 19/01/15 15:31
 */
public class MyPreception {

    /**
     * 权重W
     */
    private double[] weight;

    /**
     * 学习速率
     */
    private double rate;

    @Test
    public void lineTest() {
        double[][] x = new double[][]{
                {4, 3, 1},
                {5, 4, 1},
                {4, 5, 1},
                {1, 1, 1},
                {2, 1, 1},
                {3, 2, 1}};
        int[] d = new int[]{1, 1, 1, -1, -1, -1};
        train(0.1, x, d, 500);
        System.out.println(sgn(new double[]{1, 3, 1}));
        System.out.println(sgn(new double[]{3, 1, 1}));
    }

    @Test
    public void orTest() {
        double[][] x = new double[][]{
                {0, 0, 1},
                {0, 1, 1},
                {1, 0, 1},
                {1, 1, 1}};
        int[] d = new int[]{0, 1, 1, 1};
        train(0.1, x, d, 10000);
        System.out.println(sgn(new double[]{1, 0, 1}));
        System.out.println(sgn(new double[]{1, 1, 1}));
        System.out.println(sgn(new double[]{0, 1, 1}));
        System.out.println(sgn(new double[]{0, 0, 1}));
    }
    @Test
    public void andTest() {
        double[][] x = new double[][]{
                {0, 0, 1},
                {0, 1, 1},
                {1, 0, 1},
                {1, 1, 1}};
        int[] d = new int[]{0, 0, 0, 1};
        train(0.3, x, d, 1000);
        System.out.println(sgn(new double[]{1, 0, 1}));
        System.out.println(sgn(new double[]{1, 1, 1}));
        System.out.println(sgn(new double[]{0, 1, 1}));
        System.out.println(sgn(new double[]{0, 0, 1}));
    }

    public MyPreception() {

    }

    /**
     * 训练
     * @param x
     * @param result
     */
    public void train(double rate, double[][] x, int[] result, int times) {
        this.rate = rate;
        randomWeight(x);
        for (int length = 0; length < times; length++) {
            double[] o = new double[x.length];
            for (int i = 0; i < o.length; i++) {
                o[i] = sgn(x[i]);
            }
            for (int i = 0; i < o.length; i++) {
                for (int j = 0; j < weight.length; j++) {
                    weight[j] += rate * (result[i] - o[i]) * x[i][j];
                }
            }
        }
        System.out.println(Arrays.toString(weight));
    }

    /**
     * 转移函数
     * @param x
     * @return
     */
    private int sgn(double[] x) {
        double result = 0;
        for (int i = 0; i < x.length; i++) {
            result += x[i] * weight[i];
        }
        return result >= 0 ? 1 : -1;
    }

    /**
     * 随机初始权重
     * @param input
     */
    private void randomWeight(double[][] input) {
        this.weight = new double[input[0].length];
        for (int i = 0; i < input[0].length; i++) {
            weight[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        System.out.println(Arrays.toString(weight));
    }



    public double[] getWeight() {
        return weight;
    }

    public void setWeight(double[] weight) {
        this.weight = weight;
    }

    public double getRate() {
        return rate;
    }

    public void setRate(double rate) {
        this.rate = rate;
    }

    @Override
    public String toString() {
        return "MyPreception{" +
                "weight=" + Arrays.toString(weight) +
                ", rate=" + rate +
                '}';
    }
}
