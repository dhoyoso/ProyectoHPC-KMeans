#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import re
from time import time
from collections import defaultdict


def distancia(archivoA, archivoB):
    '''
    Este metodo calcula y devuelve la distancia entre 2 archivos usando el algoritmo de similitud de jaccard.(Basado en la explicación de http://techinpink.com/ )
    :param archivoA: 1er archivo para calcular distancia. (String)
    :param archivoB: 2ndo archivo para calcular distancia.(String)
    :return: sumaDeFrecuenciasEnLaInterseccion / (len(frequenciasAB) * 1.0): es la distancia entre los 2 archivos. (entero entre 0 y 1). y se multiplica por 1.0 para tener el valor flotante.
    '''

    stopwords = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi",
                 "ahora", "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí",
                 "alrededor", "ambos", "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas",
                 "aproximadamente", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla",
                 "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun",
                 "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas",
                 "bueno", "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
                 "claro", "comentó", "como", "con", "conmigo", "conocer", "conseguimos", "conseguir", "considera",
                 "consideró", "consigo", "consigue", "consiguen", "consigues", "contigo", "contra", "cosas", "creo",
                 "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuenta", "cuál",
                 "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar",
                 "de", "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás",
                 "dentro", "deprisa", "desde", "despacio", "despues", "después", "detras", "detrás", "dia", "dias",
                 "dice", "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
                 "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
                 "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra",
                 "enfrente", "enseguida", "entonces", "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa",
                 "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estados", "estais", "estamos",
                 "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
                 "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron",
                 "fui", "fuimos", "g", "general", "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla",
                 "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo",
                 "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos",
                 "hicieron", "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó",
                 "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "j",
                 "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar",
                 "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante",
                 "medio", "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis",
                 "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho", "muchos", "muy",
                 "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas",
                 "ninguno", "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
                 "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro",
                 "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada", "pasado", "paìs", "peor", "pero",
                 "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
                 "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque",
                 "posible", "primer", "primera", "primero", "primeros", "principalmente", "pronto", "propia", "propias",
                 "propio", "propios", "proximo", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
                 "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
                 "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó",
                 "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se",
                 "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser", "sera", "será", "serán", "sería",
                 "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
                 "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su",
                 "supuesto", "sus", "suya", "suyas", "suyo", "sé", "sí", "sólo", "t", "tal", "tambien", "también",
                 "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis", "tenemos",
                 "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda",
                 "todas", "todavia", "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos",
                 "trabajan", "trabajar", "trabajas", "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuvo",
                 "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos", "usa",
                 "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor",
                 "vamos", "van", "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
                 "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y", "ya", "yo",
                 "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
                 "último", "últimos", "a", "a's", "able", "about", "above", "according", "accordingly", "across",
                 "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                 "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and",
                 "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart",
                 "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking",
                 "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become",
                 "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
                 "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
                 "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes",
                 "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering",
                 "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                 "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                 "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
                 "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
                 "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
                 "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
                 "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
                 "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly",
                 "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
                 "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself",
                 "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                 "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
                 "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                 "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately",
                 "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
                 "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean",
                 "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself",
                 "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
                 "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
                 "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
                 "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise",
                 "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular",
                 "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
                 "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                 "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say",
                 "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                 "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
                 "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
                 "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
                 "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                 "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them",
                 "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein",
                 "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third",
                 "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                 "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
                 "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
                 "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various",
                 "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll",
                 "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when",
                 "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                 "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom",
                 "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would",
                 "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves", "z", "zero"}

    archA = mapaDeArreglosDocs[archivoA]
    archB = mapaDeArreglosDocs[archivoB]

    frequenciasDeA = {}

    frequenciasDeA = defaultdict(lambda: 0)

    frequenciasDeB = {}

    frequenciasDeB = defaultdict(lambda: 0)

    frequenciasAB = list(set(archA + archB))

    for i in range(len(archA)):
        frequenciasDeA[archA[i]] += 1

    for j in range(len(archB)):
        frequenciasDeB[archB[j]] += 1

    interseccion = list(set(set(archA) & set(archB)))

    sumaDeFrecuenciasEnLaInterseccion = 0

    for k in range(len(interseccion)):
        sumaDeFrecuenciasEnLaInterseccion = sumaDeFrecuenciasEnLaInterseccion + frequenciasDeA[interseccion[k]] + frequenciasDeB[interseccion[k]]

    return sumaDeFrecuenciasEnLaInterseccion / (len(frequenciasAB) * 1.0)

def documentosvscentroides(centroides, documentos, sinrelacion):
    '''
    Este metodo calcula las distancias que cada documento tiene con los centroides y a partir de esto genera el mapa de
    clusters.
    :param centroides: arreglo de centroides.
    :param documentos: arreglo de documentos a comparar contra los centroides (Strings).
    :param sinrelacion: set de documentos que no tienen relacion con ningun centroide.
    :return: mapaclusters: es el mapa que contiene como clave el numero(entero) de cluster y como valor un arreglo
    de documentos (strings) que pertenecen al cluster.
    '''
    comparaciones = {}

    comparaciones = defaultdict(lambda: [], comparaciones)

    # se mete en un mapa de clave:documento y valor:arreglo de K posiciones las distancias que cada doc tiene con cada K.

    for cent in centroides:
        documentos.remove(cent)

    for doc in documentos:
        for centroid in centroides:
            comparaciones[doc].append(distancia(doc,centroid))


    mapaclusters={}

    mapaclusters = defaultdict(lambda: [], mapaclusters)

    #se inizializa el valor del mapa de clusters en un arreglo vacio para hacerle .append; la clave va a ser el # de cluster.

    for initcluster in range(0,len(centroides)):
        mapaclusters[initcluster]=[]

    # se rellena el mapa de clusters teniendo en cuenta que la clave es el # de cluster que corresponde al indice del
    # valor maximo en el arreglo de comparaciones en el mapa de comparaciones. a partir de esto se sabe cual es el doc
    # que se va a meter en determinado cluster.

    for doc in documentos:
        if max(comparaciones[doc]) != 0:
            if doc in sinrelacion:
                sinrelacion.remove(doc)
            mapaclusters[comparaciones[doc].index(max(comparaciones[doc]))].append(doc)
        else:
            sinrelacion.add(doc)


    for cent in centroides:
        mapaclusters[centroides.index(cent)].append(cent)
        documentos.append(cent)

    return mapaclusters

def recentrarcentroide (mapaclusters,centroides):
    '''
    Este metodo se encarga de basado en el mapa de clusters sacar un nuevo centroide por cluster.
    :param mapaclusters:es el mapa que contiene como clave el numero(entero) de cluster y como valor un arreglo
    de documentos (strings) que pertenecen al cluster.
    :param centroides: arreglo de centroides viejos.
    :return: centroides: nuevos centroides.
    '''
    promedios={}
    promedios = defaultdict(lambda: [], promedios)
    temporal=[]

    # crea una matriz por cada cluster la cual contiene todas las distancias de docsxdocs
    # y recorre por columnas sumando y dividiendo por el total para saber cual es el
    # documento con menos distancia en promedio del cluster asignandolo asi a los nuevos
    # centroides.

    for cent in centroides:

        distancias = [[1 for x in range(0,len(mapaclusters[centroides.index(cent)]))] for y in range(0,len(mapaclusters[centroides.index(cent)]))]
        for y in range(0,len(mapaclusters[centroides.index(cent)])):
            for x in range(y+1,len(mapaclusters[centroides.index(cent)])):
                aux1=mapaclusters[centroides.index(cent)][x]
                aux2=mapaclusters[centroides.index(cent)][y]
                distancias[x][y] = distancia(aux1,aux2)
                distancias[y][x] = distancias[x][y]
        for j in range(0, len(distancias[0])):
            temporal = np.cumsum(distancias[j])
            promedios[centroides.index(cent)].append((temporal[len(temporal) - 1]) / len(temporal))



    for centroid in centroides:
        centroides[centroides.index(centroid)] = mapaclusters[centroides.index(centroid)][promedios[centroides.index(centroid)].index(max(promedios[centroides.index(centroid)]))]

    return centroides

def sacarMapaDeArreglos(documents,ruta):
    '''
    Este metodo se encarga de abrir los documentos y calcular su arreglo de palabras (frecuencias) poniendolos en un mapa
    de tipo key:document(string) value:Array de frecuencias. (Este metodo fue basado en las respuestas de https://stackoverflow.com/questions/18135967/creating-a-list-of-every-word-from-a-text-file-without-spaces-punctuation )
    :param documents: arreglo de documentos (Strings) a partir de los cuales se va a calcular los arreglos de frecuencias y
    el mapa.
    :param ruta: la ruta en donde estan estos documentos.
    :return: mapa: el mapa que contiene los arreglos de los documentos asi: key:document(string) value:Array de frecuencias.
    '''
    stopwords = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi",
                 "ahora", "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí",
                 "alrededor", "ambos", "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas",
                 "aproximadamente", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla",
                 "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun",
                 "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas",
                 "bueno", "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
                 "claro", "comentó", "como", "con", "conmigo", "conocer", "conseguimos", "conseguir", "considera",
                 "consideró", "consigo", "consigue", "consiguen", "consigues", "contigo", "contra", "cosas", "creo",
                 "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuenta", "cuál",
                 "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar",
                 "de", "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás",
                 "dentro", "deprisa", "desde", "despacio", "despues", "después", "detras", "detrás", "dia", "dias",
                 "dice", "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
                 "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
                 "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra",
                 "enfrente", "enseguida", "entonces", "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa",
                 "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estados", "estais", "estamos",
                 "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
                 "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron",
                 "fui", "fuimos", "g", "general", "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla",
                 "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo",
                 "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos",
                 "hicieron", "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó",
                 "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "j",
                 "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar",
                 "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante",
                 "medio", "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis",
                 "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho", "muchos", "muy",
                 "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas",
                 "ninguno", "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
                 "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro",
                 "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada", "pasado", "paìs", "peor", "pero",
                 "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
                 "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque",
                 "posible", "primer", "primera", "primero", "primeros", "principalmente", "pronto", "propia", "propias",
                 "propio", "propios", "proximo", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
                 "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
                 "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó",
                 "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se",
                 "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser", "sera", "será", "serán", "sería",
                 "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
                 "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su",
                 "supuesto", "sus", "suya", "suyas", "suyo", "sé", "sí", "sólo", "t", "tal", "tambien", "también",
                 "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis", "tenemos",
                 "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda",
                 "todas", "todavia", "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos",
                 "trabajan", "trabajar", "trabajas", "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuvo",
                 "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos", "usa",
                 "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor",
                 "vamos", "van", "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
                 "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y", "ya", "yo",
                 "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
                 "último", "últimos", "a", "a's", "able", "about", "above", "according", "accordingly", "across",
                 "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                 "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and",
                 "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart",
                 "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking",
                 "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become",
                 "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
                 "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
                 "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes",
                 "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering",
                 "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                 "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                 "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
                 "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
                 "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
                 "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
                 "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
                 "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly",
                 "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
                 "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself",
                 "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                 "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
                 "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                 "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately",
                 "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
                 "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean",
                 "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself",
                 "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
                 "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
                 "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
                 "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise",
                 "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular",
                 "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
                 "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                 "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say",
                 "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                 "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
                 "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
                 "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
                 "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                 "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them",
                 "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein",
                 "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third",
                 "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                 "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
                 "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
                 "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various",
                 "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll",
                 "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when",
                 "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                 "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom",
                 "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would",
                 "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves", "z", "zero"}


    mapa = {}

    for doc in documents:
        with open(ruta + doc, 'r') as file:
            text = file.read().lower()
            file.close()
            text = re.sub('[^a-z\ \']+', " ", text)
            Arch = list(text.split())

        mapa[doc] = [word for word in Arch if word not in stopwords]

    return mapa


t1 = time()
k = 4 # esta es la cantidad de clusters que se van a generar.
ruta = "/home/dhoyoso/Downloads/dataset/" #ruta de la cual se van a extraer los documentos


sinrelacion = set([]) # este es el set que contiene los archivos que no tienen ninguna relacion con ningun centroide.
mapaDeArreglosDocs = {}

# se listan todos los documentos a clusterizar y a partir de estos se saca un mapa de arreglos con las palabras que
# cada uno de estos contiene.
documentos = os.listdir(ruta)
mapaDeArreglosDocs = sacarMapaDeArreglos(documentos,ruta)

# se verifica que k>= que la cantidad de documentos porque si no no tiene sentido alguno.
assert len(documentos) >= k
# se sacan los centroides al azar.
centroides = random.sample(documentos, k)
print "centroides: ",centroides
# se saca el mapa de clusters.
mapaclusters = documentosvscentroides(centroides,documentos,sinrelacion)
print "sin relación: ",sinrelacion
print "clusters: ",mapaclusters
# se buscan nuevos centroides a partir de los clusters que ya se sacaron.
centroides = recentrarcentroide(mapaclusters,centroides)
# se calcula el nuevo mapa de cluster a partir de los nuevos centroides.
mapaclusters = documentosvscentroides(centroides,documentos,sinrelacion)

t2 = time()

print "centroides: ",centroides
print "sin relación: ",sinrelacion
print "clusters: ",mapaclusters
print "time ", t2-t1
