(ns mario-net.core
  (:require [clojure.java.io :as io]
            [taoensso.nippy :as nippy]
            [cortex.protocols :as cp]
            [cortex.backends :as b]
            [cortex.impl.layers :as impl]
            [clojure.core.matrix :as m]
            [cortex.description :as desc]
            [cortex.network :as net]
            [cortex.registry :as reg]
            [cortex.optimise :as opt]
            [mikera.image.core :as i])
  (:import [java.io ByteArrayOutputStream]))

(defn read-nippy [path]
  (with-open [input (io/input-stream path)
              output (ByteArrayOutputStream.)]
    (io/copy input output)
    (nippy/thaw (.toByteArray output))))

(defn write-nippy [path body]
  (with-open [w (io/output-stream path)]
    (.write w (nippy/freeze body))))

(defn read-edn
  [path]
  (read-string (slurp path)))

(defn edn->nippy
  [path]
  (let [new-path (clojure.string/replace path #".edn$" ".nippy")]
    (write-nippy new-path (read-edn path))
    :ok))

(defn get-dataset
  []
  (read-nippy "resources/data/training-data.nippy"))


(defrecord MegaSoftmax [softmax-shapes]
  cp/PModule
  (calc [this input]
    (let [output (or (:output this)
                     (b/new-array (m/shape input)))]
      (reduce (fn [offset n-items]
                (let [local-in (m/subvector input offset n-items)
                      local-out (m/subvector output offset n-items)
                      offset (+ offset n-items)]
                  (impl/softmax-forward! local-in local-out)
                  offset))
              0
              softmax-shapes)
      (assoc this :output output)))

  (output [this]
    (:output this))

  cp/PNeuralTraining
  (forward [this input]
    (cp/calc this input))


  (backward [this input output-gradient]
     (let [input-gradient (or (:input-gradient this)
                              (b/new-array (m/shape input)))]
       (m/assign! input-gradient output-gradient)
       (assoc this :input-gradient input-gradient)))

  (input-gradient [this]
    (:input-gradient this)))

;;(reg/register-module mario-net.core.MegaSoftmax)


(defn mega-softmax [shapes] {:type :mega-softmax :shapes shapes})


(defmethod desc/build-desc :mega-softmax
  [previous item]
  (let [io-size (:output-size previous)]
    (assoc item :input-size io-size :output-size io-size)))


(defmethod desc/create-module :mega-softmax
  [desc]
  (->MegaSoftmax (:shapes desc)))



;; (def network-desc
;;   [(desc/input 9)
;;    (desc/linear->relu 100)
;;    (desc/linear->relu 100)
;;    (desc/linear 81)
;;    (mega-softmax (vec (repeat 9 9)))])

(def network-desc
  [(desc/input 9)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear->relu 20)
   (desc/linear 81)
   (mega-softmax (vec (repeat 9 9)))])


(defn run-single-input
  [input]
  (let [network (desc/build-and-create-network network-desc)]
    (net/run network [input])))

(defn produce-answer
  [v]
  (let [n (count v)]
    (vec (mapcat (fn [e]
                   (assoc (vec (repeat n 0)) e 1))
                 v))))

(defn produce-noised-input
  [v]
  (vec (map (fn [e] (if (< 0.5 (rand))
                      -1 e)) v)))


(defn train-network
  [input-seq]
  (let [training-labels (mapv produce-answer input-seq)
        training-data (mapv produce-noised-input input-seq)
        network (desc/build-and-create-network network-desc)
        loss-fn (opt/mse-loss)
        optimizer (opt/adadelta-optimiser (cp/parameter-count network))
        _ (println "STARTING")
        network (net/train-until-error-stabilizes network optimizer loss-fn
                                                  training-data training-labels
                                                  10
                                                  training-data training-labels)]
    network))

(defn serialize-net
  [net path]
  (cortex.serialization/write-network! net (clojure.java.io/output-stream path)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Draw pictures!
(defn get-answer-from-dist
  [v]
  (let [r (rand)
        v (vec v)]
    (loop [i 0]
      (let [sum (reduce + (take (inc i) v))]
        (if (> sum r)
          i
          (recur (inc i)))))))

(defn get-picture
  [net w h]
  (let [pic (m/new-array [w h])
        points (sort-by (fn [[x y]]
                          (let [dx (- (/ w 2) x)
                                dy (- (/ w 2) y)]
                            (+ (* dx dx) (* dy dy))))
                        (for [x (range (- w 2))
                              y (range (- h 2))]
                          [x y]))]
    (m/mset! pic -1.0)
    (doseq [[x y] points]
      (let [s (m/submatrix pic x 3 y 3)
            v (m/as-vector s)
            thought (partition 9 (m/eseq (first (net/run net [v]))))
            answers (mapv get-answer-from-dist thought)]
        (doseq [i (range 9)]
          (let [new-value (nth answers i)
                x (+ x (mod i 3))
                y (+ y (quot i 3))]
            (if (or (= -1.0 (m/mget pic x y))
                    (< (rand) 0.1)
                    )
              (m/mset! pic x y (nth answers i)))))))
    pic))

(defn pic->image
  [pic]
  (let [[w h] (m/shape pic)
        index (into {} (map (comp vec reverse)
                            (read-string (slurp "resources/data/index.edn"))))
        img (i/new-image w h)
        pixels (i/get-pixels img)
        pic-values (m/as-vector pic)]
    (dotimes [i (* w h)]
      (aset pixels i (get index (Math/round (m/mget pic-values i)))))
    (i/set-pixels img pixels)
    img))

;; (m/set-current-implementation :vectorz)
;; (def loaded-net (cortex.serialization/read-network! (clojure.java.io/input-stream "mario-net.bin")))

(defn make-n-pictures
  [run-name net n w h]
  (m/set-current-implementation :vectorz)
  (dotimes [i n]
    (let [name (str "mario-net-" run-name "-" i ".png")]
      (i/save (pic->image (get-picture net w h)) name)
      (println name))))
