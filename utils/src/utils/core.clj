(ns utils.core
  (:require [clojure.string :as str]
            [clojure.tools.cli :refer [parse-opts]]
            [clojure.pprint :refer [pprint]]
            [clojure.java.io :as io])
  (:import [java.io File])
  (:gen-class))

(def ^:dynamic *debug* false)

(defn update-config [in-file parameters & {:keys [debug out show-content]
                                           :or {debug *debug*
                                                out   nil
                                                show-content false}}]
  (let [para  (mapv #(str/split % #"=") (str/split parameters #",[\s]*"))
        para_count (count para)
        lines (str/split (slurp in-file) #"[\n]")
        new-contents (mapv (fn [lin]
                             (loop [i (int 0)]
                               (if (< i para_count)
                                 (if (.contains lin ((para i) 0))
                                   (let [regexp-str (str/replace ((para i) 0) "[]" "\\[\\]")
                                         pattern (re-pattern (str/join ["\\s" regexp-str "[\\s]*=[^;]+;"]))]
                                     (when debug
                                       (pprint lin))
                                     (str/replace lin pattern
                                                  (str/join [" " ((para i) 0) " = " ((para i) 1) ";"])))
                                   (recur (unchecked-inc i)))
                                 lin)))
                           lines)]
    (when debug
      (pprint para)
      (pprint para_count))
    (when show-content
      (pprint new-contents))
    (if out
      (spit out (str/join "\n" (conj new-contents "\n")))
      (pprint (conj new-contents "\n")))))

;; (update-config "../include/pCT_config.h" "DEBUG_TEXT_ON=false,MODIFY_MLP=false")

(defn usage
  ([summary]
   (println "summary:")
   (println summary))
  ([]
   (println
    (->> ["pCT utils. Usage\n"
          "   :config      update configuration in header files\n"]
         (str/join)))))

(defn exit [status msg]
  (println msg)
  (System/exit status)
  )

(def config-cli-options
  [[nil "--header HEADER_FILE" "header configuration"
    :id :header
    :default nil
    :parse-fn #(File. %)
    :validate [#(and (.exists %) (.isFile %)) "File must exists."]]
   ["-o" "--option PCT_OPTIONS" "pCT settings"
    :id :pct-options
    :default []]
   ;; ["-l" "--list" "list available options"]
   ["-t" "--try" "Dry run/test (optional)"
    :id :try
    :default false]
   ["-f" "--file FILENAME" "Output headers path (optional)"
    :id :file-out
    :default nil]
   ["-h" "--help" "help"]])

(defn -main [& args]
  (let [mode (first args)]
    (case mode
      ":config" (let [{:keys [options arguments errors summary]} (parse-opts (rest args) config-cli-options)
                      {:keys [header pct-options try]}  options]
                  (when *debug*
                    (println)
                    (pprint "***** Debug Output *****")
                    (pprint options)
                    (pprint arguments)
                    (pprint errors)
                    (pprint pct-options)
                    (pprint "^^^^^ Debug Done ^^^^^")
                    (println))
                  (cond
                    (:help options) (exit 0 (usage summary))
                    errors (exit 1 (str/join ["Error:\n" (str/join errors)]))
                    (or (empty? pct-options)
                        (nil? header))  (exit 1 (usage summary))
                    :else (let [header-path (.getPath header)]
                            (if (:try options)
                              (update-config header-path pct-options :out (:file-out options))
                              (update-config header-path pct-options :out header-path)))))
     (usage))))

