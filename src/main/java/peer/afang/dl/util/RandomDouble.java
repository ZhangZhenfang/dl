package peer.afang.dl.util;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/3/1 22:00
 */
public class RandomDouble {
    public static double[] random(int length) {
        double[] data = new double[length];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        return data;
    }
}
