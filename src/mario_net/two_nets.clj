(ns mario-net.two-nets
  (:require [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [mikera.image.core :as i]
            [cortex.description :as desc]
            [cortex.network :as net]))

(defn index->label
  [idx c]
  (assoc (vec (repeat c 0))
         idx 1))

(defn build-dataset
  [x-offset y-offset]
  (let [d (->> "data/16x16-map.edn"
               io/resource
               slurp
               read-string)
        lowest-sprite-index (apply min (map second d))
        highest-sprite-index (apply max (map second d))
        sprite-count (inc (apply max (map second d)))
        max-y (apply max (map second (keys d)))]
    (println "lowest-sprite-index:" lowest-sprite-index)
    (println "highest-sprite-index:" highest-sprite-index)
    (println "Sprite count:" sprite-count)
    (println "Max-y:" max-y)
    (->> d
         (map (fn [[[x y] i]]
                (if-let [o (get d [(+ x x-offset) (+ y y-offset)])]
                  [(into (index->label i sprite-count) [(double (/ y max-y))])
                   (index->label o sprite-count)])))
         (filter identity))))

(defn load-index
  []
  (->> "data/16x16-index.edn"
       io/resource
       slurp
       read-string))

(def network-desc
  [(desc/input 34)
   (desc/linear->relu 100)
   (desc/linear->softmax 33)])

(defn train-network
  [x-offset y-offset]
  (let [dataset (build-dataset x-offset y-offset)
        training-data (mapv first dataset)
        training-labels (mapv second dataset)
        network (desc/build-and-create-network network-desc)]
    (println "Training network:" (count training-data) "inputs...")
    (net/train-until-error-stabilizes network training-data training-labels)))

(comment
  (def x-net (train-network 1 0))
  (def y-net (train-network 0 1))
  )

(defn generate-input
  [sprite-index y]
  (into (index->label sprite-index 33) [(double (/ y 14))]))

(defn sample-index-from
  [v]
  (let [m (reduce + v)
        r (* m (rand))]
    (loop [i 0]
      (let [sum (reduce + (take (inc i) v))]
        (if (> sum r)
          i
          (recur (inc i)))))))

(defn predict
  [x-net y-net left-index above-index y]
  (sample-index-from (seq (m/add (first (net/run x-net [(generate-input left-index y)]))
                                 (first (net/run y-net [(generate-input above-index y)]))))))

(defn nets->picture
  [x-net y-net]
  (->> (for [x (range 150)
             y (range 15)]
         [x y])
       (reduce (fn [eax [x y]]
                 (let [prediction (predict x-net y-net
                                           (get eax [(dec x) y] 13)
                                           (get eax [x (dec y)] 13)
                                           y)]
                   (assoc eax [x y] prediction)))
               {})))

(defn picture->image
  [pic]
  (let [[w h] [150 15]
        index (read-string (slurp "resources/data/16x16-index.edn"))
        img (i/new-image (* 16 w) (* 16 h))]
    (doseq [x (range w)
            y (range h)]
      (let [sub-image (i/sub-image img (* 16 x) (* 16 y) 16 16)]
        (i/set-pixels sub-image (int-array (get index (get pic [x y]))))))
    img))

(defn net->sorted-index-predictions
  [net sprite-idx y]
  (->> (seq (first (net/run net [(into (index->label sprite-idx 33) [(double (/ y 14))])])))
       (map-indexed vector)
       (sort-by second >)
       (mapv first)))

(defn net->verification-picture
  [net sprite-idx]
  (vec
   (for [y (range 15)]
     (->> (net->sorted-index-predictions net sprite-idx y)
          (concat [sprite-idx])
          (vec)))))

(defn verification-picture->img
  [pic]
  (let [[w h] [(count (first pic)) 15]
        index (read-string (slurp "resources/data/16x16-index.edn"))
        img (i/new-image (* 16 w) (* 16 h))]
    (doseq [x (range w)
            y (range h)]
      (let [sub-image (i/sub-image img (* 16 x) (* 16 y) 16 16)]
        (i/set-pixels sub-image (int-array (get index (get-in pic [y x]))))))
    img))

(defn index-image
  []
  (let [[w h] [33 1]
        index (read-string (slurp "resources/data/16x16-index.edn"))
        img (i/new-image (* 16 w) (* 16 h))]
    (doseq [x (range w)]
      (let [sub-image (i/sub-image img (* 16 x) 0 16 16)]
        (i/set-pixels sub-image (int-array (get index x)))))
    img))
