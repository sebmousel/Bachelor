% This is a LaTeX class that helps writing the master's thesis within the computer science faculty of the University of Vienna.
% It generates the title page and creates commands to easily adjust it.
% Additionally it includes the packages needed to make a PDF-A compliance possible.

% Change log:
% Created 31.01.2019    Christoph Loitzenbauer
% Edit 04.02.2019       Christoph Loitzenbauer      Created commands for multiple line titles and names
% Edit 12.02.2019       Christoph Loitzenbauer      Updated comments
% Edit 22.06.2021       Christoph Loitzenbauer      created new version because of Tex Live 2020

% -- Creating the class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{UniVieCS_Thesis}[2021/06/22 UniVie Computer Science template for the Master's Thesis]
%\LoadClass{article} %Caution: the titlepage only fits if this class is used.
%Alternatively the following class should be possible if Din a4 is a requirement.
%caution this has not been tested extensively
%replace the LoadClass{article} it with:
%\LoadClass[a4paper]{article}
%\LoadClass[a4paper, fontsize=11pt,listof=totoc,bibliography=totoc]{scrbook}
\LoadClass[footinclude=false,fontsize=11pt,paper=a4,listof=totoc,bibliography=totoc,BCOR=12mm,DIV=12]{scrbook} % two-sided % original
%\LoadClass[footinclude=false,oneside,fontsize=11pt,paper=a4,listof=totoc,bibliography=totoc,DIV=12]{scrbook} % one-sided


% -- Packages needed in the class
\RequirePackage[a-1b]{pdfx}
\RequirePackage{graphicx} % Required for including pictures
\RequirePackage[pdfa]{hyperref}
\RequirePackage[T1]{fontenc} % Required for accented characters
\RequirePackage{changepage}
\RequirePackage[final]{microtype}
\RequirePackage{xurl}
\RequirePackage{geometry}
\usepackage{layout}
\hypersetup{
	pdfpagelayout   = TwoPageRight   
}





% -- Commands for the title page --------------

\newcommand{\TitleSetup}{MASTERARBEIT / MASTER'S THESIS}
\newcommand{\TitleTitleSetup}{Titel der Masterarbeit / Title of the Master's Thesis\par}




% -- Title of the work 
\newcommand{\TitleValue}{Titel \vspace*{\baselineskip} \vspace*{\baselineskip}}

\newcommand{\SubTitleValue}{Subtitel \vspace*{\baselineskip} \vspace*{\baselineskip}}

% -- Name
\newcommand{\WhoValue}{\textgreater Akademische(r) Grad(e) Vorname Zuname Akademische(r) Grad(e)\textless \par \textgreater degree(s) first name family name degree(s)\textless \par }

% -- Desired degree
\newcommand{\DegreeValue}{Master \textgreater Zusatz\textless \hspace{0.1cm}(M \textgreater Abk\"urzung\textless) \par Master \textgreater affix\textless \hspace{0.1cm}(M \textgreater abbr.\textless) \par }

% -- Year
\newcommand{\YearGerValue}{\textgreater Jahr\textless }
\newcommand{\YearEngValue}{\textgreater Year\textless }

% -- Degree programme code
\newcommand{\ProgrammeCodeValue}{ \textgreater 000 000\textless  }

% -- Degree programme name
\newcommand{\ProgrammeNameValue}{  }

% -- Supervisors
\newcommand{\SupervisorValue}{\textgreater Akademische(r) Grad(e)  Vorname Zuname\textless \hspace{0.1cm}/}
\newcommand{\SupervisorEngValue}{ \textgreater degree(s) first name family name\textless}
\newcommand{\CoSupervisorValue}{\textgreater Akademische(r) Grad(e)  Vorname Zuname\textless \hspace{0.1cm}/}
\newcommand{\CoSupervisorEngValue}{ \textgreater degree(s) first name family name\textless}
\newcommand{\CoSupValue}{1} % ture false if there is a cosupervisor or not

% -- Commands to update the values
\newcommand{\Title}[1]{\renewcommand{\TitleValue}{#1 \vspace*{\baselineskip} \vspace*{\baselineskip} }}
\newcommand{\TitleTwo}[1]{\renewcommand{\TitleValue}{#1\grqq \vspace*{\baselineskip} }}
\newcommand{\TitleThree}[1]{\renewcommand{\TitleValue}{#1\grqq}}

\newcommand{\SubTitle}[1]{\renewcommand{\SubTitleValue}{#1 \vspace*{\baselineskip} \vspace*{\baselineskip} }}


\newcommand{\Who}[1]{\renewcommand{\WhoValue}{#1 \vspace*{\baselineskip}}}
\newcommand{\WhoTwo}[1]{\renewcommand{\WhoValue}{#1 }}
\newcommand{\Degree}[1]{\renewcommand{\DegreeValue}{#1 \vspace*{\baselineskip} \vspace*{\baselineskip}} }
\newcommand{\YearGer}[1]{\renewcommand{\YearGerValue}{#1}}
\newcommand{\YearEng}[1]{\renewcommand{\YearEngValue}{#1}}
\newcommand{\ProgrammeCode}[1]{\renewcommand{\ProgrammeCodeValue}{#1}}
\newcommand{\ProgrammeName}[1]{\renewcommand{\ProgrammeNameValue}{#1}}
\newcommand{\Supervisor}[1]{\renewcommand{\SupervisorValue}{#1} \renewcommand{\SupervisorEngValue}{}}
\newcommand{\CoSupervisor}[1]{\renewcommand{\CoSupervisorValue}{#1}\renewcommand{\CoSupervisorEngValue}{}}
\newcommand{\SupervisorTwo}[1]{\renewcommand{\SupervisorEngValue}{#1}}
\newcommand{\CoSupervisorTwo}[1]{\renewcommand{\CoSupervisorEngValue}{#1}}
\newcommand{\Year}[1]{\YearGer{#1} \YearEng{#1}}

% -- Command to generate the titlepage
\newcommand{\Titlepage}{
    \newgeometry{bottom=0.1cm}
	\begin{titlepage}
		\enlargethispage{5\baselineskip}
		\fontfamily{cmr}\selectfont
		\thispagestyle{empty}
		\begin{center}
            \begin{adjustwidth}{-0.5cm}{-2cm}    
				\vspace*{-2cm}
				\hspace*{10.17cm}
				%University logo
				\includegraphics[width=0.45\linewidth]{figures/univie.jpeg} \par
				\flushleft
    \vspace*{-2.7cm}
				\hspace*{-1cm}
                \includegraphics[width=0.3\linewidth]{figures/VERA.jpeg}\par
    \vspace{1cm}
			    \vspace{1cm}
				\centering 	
				
				% Title
				{\scshape{\Large\TitleSetup\par}}
				\vspace{1.2cm}
				%Thesis title
				{\Huge\TitleValue}
    
                \vspace{-1cm}
                
                {\Large\color{red}\textbf{\underline{\SubTitleValue}}\par}
                
				\vspace{1.5cm} % If title is multiple lines long change the space here
				
				
				submitted by  \vspace{0.2cm}\linebreak
				{\fontsize{13}{16}\selectfont \WhoValue}
		
				
				\vspace{1.5cm} %if name is multiple lines long change the space here
				
				in partial fulfilment of the requirements for the degree of \vspace{0.2cm}\linebreak
				{\fontsize{13}{16}\selectfont \DegreeValue \par}
				\vspace{1cm}
				
				%\flushleft
				
				
				\begin{tabular}{lcl}
					Vienna, \YearEngValue	\linebreak
					\vspace{1cm}&  &  \\
					
					Degree programme code as it appears on & & \\
					the student record sheet:  & \hspace{1cm}& UA \ProgrammeCodeValue \vspace{0.3cm} \\
					
					  
					Degree programme as it appears on & & \\
					the student record sheet: & \hspace{1cm} & \ProgrammeNameValue 
					 \vspace{0.3cm}\\
					
					Supervisor: & \hspace{1cm} & \SupervisorValue \\
					%& \hspace{1cm} & \SupervisorEngValue \\
					%\if\CoSupervisorValue0
					%\else
					%Mitbetreut von / Co-Supervisor: & \hspace{1cm} & \CoSupervisorValue \\
					%& \hspace{1cm} & \CoSupervisorEngValue \\
					%\fi
				\end{tabular}
               
                
			\end{adjustwidth}	

		\end{center}
	\end{titlepage}
	\restoregeometry
}