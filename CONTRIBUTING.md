
Contributing to bilby
=================

This is a short guide to help get you started contributing to bilby.

Getting started
-------------------

All the code lives in a git repository (for a short introduction to git, see
[this tutorial](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html))
which is hosted here: https://git.ligo.org/lscsoft/bilby.  If you haven't
already, you should
[fork](https://docs.gitlab.com/ee/gitlab-basics/fork-project.html) the repository
and clone your fork, i.e., on your local machine run

```bash
$ git clone git@git.ligo.org:albert.einstein/bilby.git
```

replacing the SSH url to that of your fork. This will create a directory `/bilby`
containing a local copy of the code. From this directory, you can run

```bash
$ python setup.py develop
```

which will install `bilby` and, because we used `develop` instead of `install`
when you change the code your installed version will automatically be updated.

---

#### Removing previously installed versions

If you have previously installed `bilby` using `pip` (or generally find buggy
behaviour). It may be worthwhile purging your system and reinstalling. To do
this, first find out where the module is being imported from: from any
directory that is *not* the source directory, do the following

```bash
$ python
>>> import bilby
>>> print(bilby.__file__)
/home/user/anaconda2/lib/python2.7/site-packages/bilby-0.2.2-py2.7.egg/bilby/__init__.pyc
```
In the example output above, we see that the directory that module is installed
in. To purge our python installation, run

```bash
$ rm -r /home/user/anaconda2/lib/python2.7/site-packages/bilby*
```

You can then verify this was successful by trying to import bilby in the python
interpreter.

Discussion
----------

If you've run into behavior you don't understand, or you're
having trouble working out a good way to apply it to your code, or
you've found a bug or would like a feature it doesn't have, we want to
hear from you!

Our main forum for discussion is the project's [GitLab issue
tracker](https://git.ligo.org/lscsoft/bilby/issues). This is the right
place to start a discussion of any of the above or most any other
topic concerning the project.

#### Code of Conduct

Everyone participating in the bilby community, and in particular in our
issue tracker, pull requests, and IRC channel, is expected to treat
other people with respect and more generally to follow the guidelines
articulated in the [Python Community Code of
Conduct](https://www.python.org/psf/codeofconduct/).


Submitting Changes
------------------------

All changes to the code base go through the [pull-request
workflow](https://docs.gitlab.com/ee/user/project/merge_requests/) Anyone may
review your code and you may expect a few comments and questions.  Once all
discussions are resolved, core developers will approve the merge request and
then merge in into master. If you go a few days without a reply, please feel
free to ping the thread by adding a new comment.

All merge requests should be focused: they should aim to either add one
feature, solve one bug, or fix some stylistic issues. If multiple changes are
lumped together it can slow down the process and make it harder to review.

Before you begin: if your change will be a significant amount of work
to write, we highly recommend starting by opening an issue laying out
what you want to do.  That lets a conversation happen early in case
other contributors disagree with what you'd like to do or have ideas
that will help you do it.

We are keen to maintain a high standard of the code. This makes it easier to
maintain, develop, and track down buggy behaviour.

All merge requests should also be recorded in the CHANGELOG.md.
This just requires a short sentence describing describing the change, e.g.,
"- Enable initial values for emcee to be specified."

ADD DISCUSSION ON CODE QUALITY

Reviewing Changes
-----------------

If you are reviewing a merge request (either as a core developer or just as an
interested party) please key these three things in mind

* If you open a discussion, be timely in responding to the submitter. Note, the
  reverse does not need to apply.
* Keep your questions/comments focussed on the scope of the merge request. If
  while reviewing the code you notice other things which could be improved, open
  a new issue.
* Be supportive - merge requests represent a lot of hard work and effort and
  should be encouraged.

Core developer guidelines
-------------------------

Core developers should follow these rules when processing pull requests:

* Always wait for tests to pass before merging PRs.
* Delete branches for merged PRs (by core devs pushing to the main repo).
* Make sure related issues are linked and (if appropriate) closed.
* Squash commits


Issue-tracker conventions
-------------------------

#### Bug reporting

If you notice any bugs in the code, please let us know!

Issues reporting bugs should include:
* A brief description of the issue.
* A full error trace (if applicable).
* A minimal example to reproduce the issue.

#### Feature requests

If there's any additional functionality you'd like to see in the code open an
issue describing the desired feature.

In order to maximise the likelihood that feature requests will be fulfilled
you should:
* Make the feature request as specific as possible, vague requests are
  unlikely to be taken on by users.
* If possible break down the feature into small chunks which can be
  marked off on a checklist.



## Code overview

In this section, we'll give an overview of how the code is structured. This is intended to help orient users and make it easier to contribute. The layout is intended to define the logic of the code and new merge requests should aim to fit within this logic (unless there is a good argument to change it). For example, code which adds a new sampler should not effect the gravitational-wave specific parts of the code. Note that this document is not programatically generated and so may get out of date with time. If you notice something wrong, please open an issue.

### Bilby Code Layout

![bilby overview](docs/images/bilby_layout.png)

Note this layout is not comprehensive, for example only a few example "Priors" are shown.
