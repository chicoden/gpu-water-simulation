export default class Matrix4 extends Float32Array {
    constructor(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33
    ) {
        super([
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33
        ]);
    }

    static identity() {
        return new Matrix4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }

    static perspProjection(focalLength, aspectRatio, zNear, zFar) {
        let zScale = zFar / (zNear - zFar);
        return new Matrix4(
            focalLength / aspectRatio, 0.0, 0.0, 0.0,
            0.0, focalLength, 0.0, 0.0,
            0.0, 0.0, zScale, -1.0,
            0.0, 0.0, zScale * zNear, 0.0
        );
    }

    getTranslationComponent() {
        return [this[12], this[13], this[14]];
    }

    translate(offset) {
        this[12] += offset[0];
        this[13] += offset[1];
        this[14] += offset[2];
    }

    yaw(angle) {
        let co = Math.cos(angle), si = Math.sin(angle);
        let oxx = this[0], oxy = this[1], oxz = this[ 2];
        let ozx = this[8], ozy = this[9], ozz = this[10];
        this[ 0] = oxx * co + ozx * si;
        this[ 1] = oxy * co + ozy * si;
        this[ 2] = oxz * co + ozz * si;
        this[ 8] = ozx * co - oxx * si;
        this[ 9] = ozy * co - oxy * si;
        this[10] = ozz * co - oxz * si;
    }

    pitch(angle) {
        let co = Math.cos(angle), si = Math.sin(angle);
        let ozx = this[8], ozy = this[9], ozz = this[10];
        let oyx = this[4], oyy = this[5], oyz = this[ 6];
        this[ 8] = ozx * co + oyx * si;
        this[ 9] = ozy * co + oyy * si;
        this[10] = ozz * co + oyz * si;
        this[ 4] = oyx * co - ozx * si;
        this[ 5] = oyy * co - ozy * si;
        this[ 6] = oyz * co - ozz * si;
    }

    roll(angle) {
        let co = Math.cos(angle), si = Math.sin(angle);
        let oxx = this[0], oxy = this[1], oxz = this[2];
        let oyx = this[4], oyy = this[5], oyz = this[6];
        this[0] = oxx * co + oyx * si;
        this[1] = oxy * co + oyy * si;
        this[2] = oxz * co + oyz * si;
        this[4] = oyx * co - oxx * si;
        this[5] = oyy * co - oxy * si;
        this[6] = oyz * co - oxz * si;
    }

    inverseOrthonormal() {
        let tx = this[0] * this[12] + this[1] * this[13] + this[ 2] * this[14];
        let ty = this[4] * this[12] + this[5] * this[13] + this[ 6] * this[14];
        let tz = this[8] * this[12] + this[9] * this[13] + this[10] * this[14];
        return new Matrix4(
            this[0], this[4], this[ 8], 0.0,
            this[1], this[5], this[ 9], 0.0,
            this[2], this[6], this[10], 0.0,
            -tx, -ty, -tz, 1.0
        );
    }

    transformDirection(v) {
        return [
            this[0] * v[0] + this[4] * v[1] + this[ 8] * v[2],
            this[1] * v[0] + this[5] * v[1] + this[ 9] * v[2],
            this[2] * v[0] + this[6] * v[1] + this[10] * v[2]
        ];
    }

    transformMatrix(other) {
        return new Matrix4(
            this[ 0] * other[ 0] + this[ 4] * other[ 1] + this[ 8] * other[ 2] + this[12] * other[ 3],
            this[ 1] * other[ 0] + this[ 5] * other[ 1] + this[ 9] * other[ 2] + this[13] * other[ 3],
            this[ 2] * other[ 0] + this[ 6] * other[ 1] + this[10] * other[ 2] + this[14] * other[ 3],
            this[ 3] * other[ 0] + this[ 7] * other[ 1] + this[11] * other[ 2] + this[15] * other[ 3],
            this[ 0] * other[ 4] + this[ 4] * other[ 5] + this[ 8] * other[ 6] + this[12] * other[ 7],
            this[ 1] * other[ 4] + this[ 5] * other[ 5] + this[ 9] * other[ 6] + this[13] * other[ 7],
            this[ 2] * other[ 4] + this[ 6] * other[ 5] + this[10] * other[ 6] + this[14] * other[ 7],
            this[ 3] * other[ 4] + this[ 7] * other[ 5] + this[11] * other[ 6] + this[15] * other[ 7],
            this[ 0] * other[ 8] + this[ 4] * other[ 9] + this[ 8] * other[10] + this[12] * other[11],
            this[ 1] * other[ 8] + this[ 5] * other[ 9] + this[ 9] * other[10] + this[13] * other[11],
            this[ 2] * other[ 8] + this[ 6] * other[ 9] + this[10] * other[10] + this[14] * other[11],
            this[ 3] * other[ 8] + this[ 7] * other[ 9] + this[11] * other[10] + this[15] * other[11],
            this[ 0] * other[12] + this[ 4] * other[13] + this[ 8] * other[14] + this[12] * other[15],
            this[ 1] * other[12] + this[ 5] * other[13] + this[ 9] * other[14] + this[13] * other[15],
            this[ 2] * other[12] + this[ 6] * other[13] + this[10] * other[14] + this[14] * other[15],
            this[ 3] * other[12] + this[ 7] * other[13] + this[11] * other[14] + this[15] * other[15]
        );
    }
}