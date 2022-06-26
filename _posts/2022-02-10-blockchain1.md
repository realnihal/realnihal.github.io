---
layout: post
title:  An Incomplete Introduction to Blockchain Part-1
description: Hashing algorithms, Immutable ledgers, Distributed P2P networks
date:   2022-02-10 15:01:35 +0300q
image:  '/img/posts/blockchain/blockchain.jpg'
tags:   [Blockchain, Guide]
---

## Time-Stamping and hashing digital documents

So far in history we have had many methods to prove the validity of time sensitive documents. Documents such as contracts, IPs, land ownership records, etc. Older popular methods include:

 1. Dated and Timed ledgers with no gaps and occasional stamps/signs.
 2. unopened self mail with the postage date.
 3. Physical evidence of a fact to predate a time

But In almost every case the only way to disprove or validate the claims is to inspect for tampering. This relies heavily on human expertise and like with everything humans do there is a scope for errors.

A **block** is essentially a way of storing time-sensitive information in a way that can be easily checked and verified with computers instead of humans. A **blockchain** is simply put a collection of many such blocks.  

This gets extremely important essentially in this day and age with more documents choosing a digital form where they can easily tampered with. What is needed is a method of time-stamping digital documents with the following two properties.
1. One must find a way to time-stamp the data itself, without any reliance on the characteristics of the medium on which the data appears, so that it is impossible to change even one bit of the document without the change being apparent. 
2. It should be impossible to stamp a document with a time and date different from the actual one.

## Hashing Algorithms

Enter the hashing algorithm, its a cryptographic function which can convert any **arbitrary** length string into  bit-strings of a **fixed** length. For example, the sha256 hashing algorithm used by the bitcoin network can convert literally anything into a 64 character, 256 bit hexadecimal string.

for any x, x_1 ; h(x) = h(x_1) is never possible. There are no duplicates.

![SHA1 vs SHA256 - KeyCDN Support](https://www.keycdn.com/img/support/sha1-vs-sha256.png)

Any small change even a **single bit** will then change the hash value completely. This is called the avalanche effect.

To learn more about Hashing functions [click here](https://webspace.science.uu.nl/~tel00101/liter/Books/CrypCont.pdf)
To learn more about the time-stamping of documents [click here](https://www.anf.es/pdf/Haber_Stornetta.pdf)
You can try a demo of hashing [here](https://tools.superdatascience.com/blockchain/hash)
## Immutable ledgers
The blockchain is officially a digital, decentralised, distributed **ledger**. You might wonder how this next-generation technology is at its core a boring book. But being an immutable ledger is the core principle/property of the blockchain.

Any information that has to be stored has to be immutable ie, cannot be changed any further. This is done to ensure that no entry will ever be tampered with.

A simple blockchain achieves this in a simple way, let me explain.

|Input|Data|
|--|--|
|previous block hash|"xyz"|
|Time|2nd July 14:03|
|Nonce|12345|
|Sender|Nihal|
|Receiver|Sudarshan|
|Amount|5 BTC|

Say this is an example block containing the transaction information of me sending 5 BTC to my friends. I have mentioned the time of the transaction and a number called the nonce(will get to later). We know take this information and pass it through the SHA-256 and mine it. Mining is essentially calculating the nonce for the data where the hash can be within a limit(will be explained in detail later in the article).
 
![Example of a Block that has been mined](/img/posts/blockchain/blockchain.png)

Now we move to the next block and create some more hashes. Since we link each block with the hash of the previous block. Even a minor change in the information of any block will lead to the change in every single hash in the entire chain. In the event this sort of change occurs the chain is rendered corrupt and the previous proper version is restored.

You can try a demo version [here](https://tools.superdatascience.com/blockchain/block)

## Distributed P2P network
"Decentralization" is one of the fundamental and is commonly used in privacy focused projects. Although it is the at the core of its application the definition and meaning is very poorly defined. Here I have made my attempt to portray the intuition behind the concept instead.

![](https://miro.medium.com/max/1094/1*WG5_xDDwHv0lMaVUYLNbVA.png)

Traditional systems, be it computer servers or banking systems all use the centralized network. All the computation, responsibility and ownership is heavily almost completely controlled by the central entity. “distributed means not all the processing of the transactions is done in the same place”, whereas “decentralized means that not one single entity has control over all the processing”. Meanwhile, the top answer on the Ethereum stack exchange gives a [very similar diagram](http://ethereum.stackexchange.com/questions/7812/question-on-the-terms-distributed-and-decentralised), but with the words “decentralized” and “distributed” switched places! Clearly, a clarification is in order.

>So here's my take on it -

 - **Architectural (de)centralization** — how many **physical computers** is a system made up of? How many of those computers can it tolerate breaking down at any single time?
- **Political (de)centralization**  — how many **individuals or organizations** ultimately control the computers that the system is made up of?
- **Logical (de)centralization**— does the **interface and data structures** that the system presents and maintains look more like a single monolithic object, or an amorphous swarm? One simple heuristic is: if you cut the system in half, including both providers and users, will both halves continue to fully operate as independent units?

### Three reasons for Decentralization

The next question is, why is decentralization useful in the first place? There are generally several arguments raised:

- **Fault tolerance**— decentralized systems are less likely to fail accidentally because they rely on many separate components that are not likely.
- **Attack resistance**— decentralized systems are more expensive to attack and destroy or manipulate because they lack  [sensitive central points](http://starwars.wikia.com/wiki/Thermal_exhaust_port)  that can be attacked at much lower cost than the economic size of the surrounding system.
- **Collusion resistance** — it is much harder for participants in decentralized systems to collude to act in ways that benefit them at the expense of other participants, whereas the leaderships of corporations and governments collude in ways that benefit themselves but harm less well-coordinated citizens, customers, employees and the general public all the time.